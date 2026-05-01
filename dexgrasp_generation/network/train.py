from hydra import compose, initialize
import logging
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from data.dataset import get_dex_dataloader
from trainer import Trainer
from utils.global_utils import log_loss_summary, add_dict
from omegaconf import OmegaConf
from omegaconf.omegaconf import open_dict
import os
from os.path import join as pjoin
from tqdm import tqdm

import argparse

from utils.interrupt_handler import InterruptHandler


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed generation training requires CUDA.")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

    return {
        "distributed": distributed,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
    }


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(cfg):
    return int(cfg.get("rank", 0)) == 0


def process_config(cfg, dist_info, save=True):
    root_dir = cfg["exp_dir"]
    os.makedirs(root_dir, exist_ok=True)

    with open_dict(cfg):
        cfg["distributed"] = dist_info["distributed"]
        cfg["rank"] = dist_info["rank"]
        cfg["local_rank"] = dist_info["local_rank"]
        cfg["world_size"] = dist_info["world_size"]
        cfg["device"] = f'cuda:{cfg["local_rank"] if cfg["distributed"] else cfg["cuda_id"]}' if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        device_id = int(cfg["local_rank"] if cfg["distributed"] else cfg["cuda_id"])
        torch.cuda.set_device(device_id)

    if save and is_main_process(cfg):
        yaml_path = pjoin(root_dir, "config.yaml")
        print(f"Saving config to {yaml_path}")
        with open(yaml_path, 'w') as f:
            print(OmegaConf.to_yaml(cfg), file=f)

    return cfg


def log_tensorboard(writer, mode, loss_dict, cnt, epoch):
    if writer is None:
        return
    for key, value in loss_dict.items():
        writer.add_scalar(mode + "/" + key, value / cnt, epoch)
    writer.flush()


def reduce_summary_dict(summary, cfg):
    if not cfg.get("distributed", False):
        return summary

    reduced = {}
    for key, value in summary.items():
        if isinstance(value, dict):
            reduced[key] = reduce_summary_dict(value, cfg)
            continue

        tensor = torch.as_tensor(value, dtype=torch.float64, device=cfg["device"])
        op = dist.ReduceOp.MAX if str(key).endswith("_max") else dist.ReduceOp.SUM
        dist.all_reduce(tensor, op=op)
        reduced[key] = tensor.cpu().numpy() if tensor.ndim > 0 else tensor.item()

    return reduced


def set_sampler_epoch(loader, epoch):
    sampler = getattr(loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def main(cfg, dist_info):
    cfg = process_config(cfg, dist_info)

    """ Logging """
    log_dir = cfg["exp_dir"]
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(f'TrainModel.rank{cfg["rank"]}')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if is_main_process(cfg):
        file_handler = logging.FileHandler(f'{log_dir}/log.txt')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        logger.addHandler(logging.NullHandler())

    """ Tensorboard """
    writer = SummaryWriter(pjoin(log_dir, "tensorboard")) if is_main_process(cfg) else None

    """ DataLoaders """
    train_loader = get_dex_dataloader(cfg, "train")
    test_loader = get_dex_dataloader(cfg, "test")

    """ Trainer """
    trainer = Trainer(cfg, logger)
    start_epoch = trainer.resume()
    trainer.enable_distributed()

    """ Test """
    def test_all(dataloader, mode, iteration):
        test_loss = {}
        for _, data in enumerate(tqdm(dataloader, disable=not is_main_process(cfg))):
            _, loss_dict = trainer.test(data)
            loss_dict["cnt"] = 1
            add_dict(test_loss, loss_dict)

        test_loss = reduce_summary_dict(test_loss, cfg)
        cnt = test_loss.pop("cnt")
        if is_main_process(cfg):
            log_loss_summary(test_loss, cnt,
                             lambda x, y: logger.info(f'{mode} {x} is {y}'))
            log_tensorboard(writer, mode, test_loss, cnt, iteration)

    """ Train """
    # Upon SIGINT, it will save the current model before exiting
    with InterruptHandler() as h:
        train_loss = {}
        for epoch in range(start_epoch, cfg["total_epoch"]):
            set_sampler_epoch(train_loader, epoch)
            set_sampler_epoch(test_loader, epoch)
            for _, data in enumerate(tqdm(train_loader, disable=not is_main_process(cfg))):
                loss_dict = trainer.update(data)
                loss_dict["cnt"] = 1
                add_dict(train_loss, loss_dict)

                if trainer.iteration % cfg["freq"]["plot"] == 0:
                    train_loss = reduce_summary_dict(train_loss, cfg)
                    cnt = train_loss.pop("cnt")
                    if is_main_process(cfg):
                        log_loss_summary(train_loss, cnt,
                                         lambda x, y: logger.info(f"Train {x} is {y}"))
                        log_tensorboard(writer, "train", train_loss, cnt, trainer.iteration)

                    train_loss = {}

                if trainer.iteration % cfg["freq"]["step_epoch"] == 0:
                    trainer.step_epoch()

                if trainer.iteration % cfg["freq"]["test"] == 0:
                    test_all(test_loader, "test", trainer.iteration)

                if trainer.iteration % cfg["freq"]["save"] == 0 and is_main_process(cfg):
                    trainer.save()

                if h.interrupted:
                    break

            if h.interrupted:
                break

    if is_main_process(cfg):
        trainer.save()
    if cfg.get("distributed", False):
        dist.barrier()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, default="ipdf_config")
    parser.add_argument("--exp-dir", type=str, help="E.g., './ipdf_train'.")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None)
    return parser.parse_known_args()


if __name__ == "__main__":
    args, overrides = parse_args()
    initialize(version_base=None, config_path="../configs", job_name="train")
    if args.exp_dir is not None:
        overrides = [f"exp_dir={args.exp_dir}", *overrides]
    cfg = compose(config_name=args.config_name, overrides=overrides)
    dist_info = setup_distributed()
    try:
        main(cfg, dist_info)
    finally:
        cleanup_distributed()
