from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import os
from os.path import join as pjoin
import sys

base_dir = os.path.dirname(__file__)
sys.path.append(pjoin(base_dir, '..'))  # data -> model -> root, to import data_proc
sys.path.append(pjoin(base_dir, '..', '..'))  # data -> model -> root, to import data_proc

from datasets.dex_dataset import DFCDataset
from datasets.object_dataset import Meshdata


def get_dex_dataloader(cfg, mode="train", shuffle=None):
    if shuffle is None:
        shuffle = (mode == "train")

    dataset = DFCDataset(cfg, mode)
    sampler = None
    if cfg.get("distributed", False):
        sampler = DistributedSampler(
            dataset,
            num_replicas=cfg["world_size"],
            rank=cfg["rank"],
            shuffle=shuffle,
        )
        shuffle = False
    return DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=shuffle, sampler=sampler, num_workers=cfg["num_workers"])

def get_mesh_dataloader(cfg, mode="train"):
    dataset = Meshdata(cfg, mode)
    sampler = None
    if cfg.get("distributed", False):
        sampler = DistributedSampler(
            dataset,
            num_replicas=cfg["world_size"],
            rank=cfg["rank"],
            shuffle=False,
        )
    return DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False, sampler=sampler, num_workers=cfg["num_workers"])
