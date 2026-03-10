#!/bin/bash
HF="$HOME/.local/bin/hf"
if [ ! -f "$HF" ]; then
  echo "FATAL: hf CLI not found at $HF"
  exit 1
fi
echo "Using: $HF"

DISKS=($(ls /dev/disk/by-id/virtio-* | grep -v e02054bd | sort))
echo "Found ${#DISKS[@]} new disks + 1 existing = $((${#DISKS[@]}+1)) total"

MODELS=(
  "Qwen/Qwen2.5-VL-3B-Instruct"
  "OpenGVLab/InternVL3_5-8B"
  "llava-hf/llava-onevision-qwen2-7b-ov-hf"
  "microsoft/Phi-4-multimodal-instruct"
  "deepseek-ai/deepseek-vl2-small"
  "openbmb/MiniCPM-V-4_5"
  "google/gemma-3-12b-it"
)

PIDS=()

# Volume 0: already mounted
echo ""
echo "=== Vol 0: /mnt/volume (already mounted) -> ${MODELS[0]} ==="
HF_HOME=/mnt/volume/huggingface $HF download "${MODELS[0]}" > /mnt/volume/download.log 2>&1 &
PIDS+=($!)
echo "  PID $!"

# Volumes 1-6: format, mount, download
for i in $(seq 0 $((${#DISKS[@]} - 1))); do
  MI=$((i + 1))
  if [ $MI -ge ${#MODELS[@]} ]; then break; fi
  DISK="${DISKS[$i]}"
  MODEL="${MODELS[$MI]}"
  MOUNT="/mnt/vol${i}"
  DISK_ID=$(basename "$DISK")

  echo ""
  echo "=== Vol $MI: $DISK_ID -> $MODEL ==="

  sudo mkfs.ext4 -F "$DISK" > /dev/null 2>&1
  sudo mkdir -p "$MOUNT"
  sudo mount "$DISK" "$MOUNT"
  sudo chown ubuntu:ubuntu "$MOUNT"
  echo "  Mounted at $MOUNT"

  HF_HOME=${MOUNT}/huggingface $HF download "$MODEL" > ${MOUNT}/download.log 2>&1 &
  PIDS+=($!)
  echo "  PID $!"
done

echo ""
echo "========================================"
echo "ALL ${#PIDS[@]} DOWNLOADS LAUNCHED"
echo "========================================"
echo "PIDs: ${PIDS[@]}"
