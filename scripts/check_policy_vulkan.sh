#!/usr/bin/env bash
set -euo pipefail

status=0
tmp_file=""

cleanup() {
  if [[ -n "$tmp_file" && -f "$tmp_file" ]]; then
    rm -f "$tmp_file"
  fi
}
trap cleanup EXIT

echo "== NVIDIA driver =="
if command -v nvidia-smi >/dev/null 2>&1; then
  if ! nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader; then
    nvidia-smi || true
    status=1
  fi
else
  echo "nvidia-smi: not found"
  status=1
fi

echo
echo "== Vulkan environment =="
printf 'VK_ICD_FILENAMES=%s\n' "${VK_ICD_FILENAMES:-<unset>}"
printf 'VK_DRIVER_FILES=%s\n' "${VK_DRIVER_FILES:-<unset>}"
printf 'NVIDIA_DRIVER_CAPABILITIES=%s\n' "${NVIDIA_DRIVER_CAPABILITIES:-<unset>}"
printf 'DISPLAY=%s\n' "${DISPLAY:-<unset>}"
printf 'XDG_RUNTIME_DIR=%s\n' "${XDG_RUNTIME_DIR:-<unset>}"

echo
echo "== Vulkan ICD files =="
icd_count=0
nvidia_icd_count=0
for dir in /usr/share/vulkan/icd.d /etc/vulkan/icd.d; do
  [[ -d "$dir" ]] || continue
  while IFS= read -r -d '' icd_file; do
    icd_count=$((icd_count + 1))
    echo "$icd_file"
    if grep -Eiq 'nvidia|libGLX_nvidia|libvulkan_nvidia' "$icd_file"; then
      nvidia_icd_count=$((nvidia_icd_count + 1))
    fi
  done < <(find "$dir" -maxdepth 1 -type f -name '*.json' -print0 2>/dev/null)
done

if [[ "$icd_count" -eq 0 ]]; then
  echo "No Vulkan ICD JSON files found under /usr/share/vulkan/icd.d or /etc/vulkan/icd.d."
  status=1
elif [[ "$nvidia_icd_count" -eq 0 ]]; then
  echo "No NVIDIA Vulkan ICD JSON found. Isaac Gym camera rendering will not see an NVIDIA device."
  status=1
fi

echo
echo "== vulkaninfo --summary =="
if command -v vulkaninfo >/dev/null 2>&1; then
  tmp_file="$(mktemp "${TMPDIR:-/tmp}/unidexgrasp-vulkaninfo.XXXXXX")"
  set +e
  vulkaninfo --summary 2>&1 | tee "$tmp_file"
  vulkaninfo_status=${PIPESTATUS[0]}
  set -e
  if [[ "$vulkaninfo_status" -ne 0 ]]; then
    echo "vulkaninfo failed with status $vulkaninfo_status."
    status=1
  elif ! grep -Eiq 'NVIDIA|deviceName[[:space:]]*=.*NVIDIA|GPU[[:space:]]+id.*NVIDIA' "$tmp_file"; then
    echo "vulkaninfo did not report an NVIDIA Vulkan device."
    status=1
  fi
else
  echo "vulkaninfo: not found. Install vulkan-tools to verify Isaac Gym camera rendering before vision smoke."
  status=1
fi

echo
if [[ "$status" -eq 0 ]]; then
  echo "Vulkan preflight passed: an NVIDIA Vulkan device is visible."
else
  echo "Vulkan preflight failed: fix the NVIDIA Vulkan/graphics driver layer before running policy vision smoke."
fi

exit "$status"
