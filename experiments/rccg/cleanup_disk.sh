#!/usr/bin/env bash
set -euo pipefail

echo "=== Disk cleanup ==="
df -h /

echo "--- Cleaning caches ---"
rm -rf /home/ubuntu/.cache/pip
rm -rf /home/ubuntu/.cache/huggingface
rm -rf /home/ubuntu/miniconda3/pkgs
rm -rf /home/ubuntu/fada-v3/data/Fetal\ Ultrasound  # remove partial extraction

echo "--- After cleanup ---"
df -h /
du -sh /home/ubuntu/miniconda3 /home/ubuntu/fada-v3 /home/ubuntu/.cache 2>/dev/null || true

echo "--- Trying to mount volume ---"
if [ -b /dev/vdc ]; then
    sudo mkdir -p /mnt/models
    sudo mount /dev/vdc /mnt/models 2>/dev/null && echo "Volume mounted" || echo "Volume mount failed"
    sudo chown ubuntu:ubuntu /mnt/models 2>/dev/null || true
    df -h /mnt/models 2>/dev/null || true
else
    echo "No /dev/vdc device found"
fi
