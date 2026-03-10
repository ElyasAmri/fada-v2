#!/bin/bash
HF="$HOME/.local/bin/hf"
TOKEN="$1"

echo "=== Downloading gemma-3-12b-it to /mnt/vol5 ==="
HF_HOME=/mnt/vol5/huggingface HF_TOKEN="$TOKEN" $HF download "google/gemma-3-12b-it" 2>&1 | tail -5
echo ""

echo "=== Downloading medgemma-4b-it to /mnt/volume ==="
# Reuse /mnt/volume - Qwen2.5-VL-3B is small, plenty of room for medgemma too
HF_HOME=/mnt/volume/huggingface HF_TOKEN="$TOKEN" $HF download "google/medgemma-4b-it" 2>&1 | tail -5
echo ""

echo "=== FINAL DISK USAGE ==="
df -h /mnt/volume /mnt/vol0 /mnt/vol1 /mnt/vol2 /mnt/vol3 /mnt/vol4 /mnt/vol5
