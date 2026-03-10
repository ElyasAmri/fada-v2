#!/bin/bash
export HF_HOME=/mnt/volume/huggingface

MODELS=(
  "Qwen/Qwen2.5-VL-3B-Instruct"
  "OpenGVLab/InternVL3_5-8B"
  "llava-hf/llava-onevision-qwen2-7b-ov-hf"
  "microsoft/Phi-4-multimodal-instruct"
  "deepseek-ai/deepseek-vl2-small"
  "openbmb/MiniCPM-V-4_5"
  "google/gemma-3-12b-it"
)

mkdir -p /mnt/volume/logs

for model in "${MODELS[@]}"; do
  SAFE=$(echo "$model" | tr "/" "_")
  echo "Starting download: $model"
  nohup ~/.local/bin/huggingface-cli download "$model" \
    > /mnt/volume/logs/${SAFE}.log 2>&1 &
done

echo "Launched ${#MODELS[@]} parallel downloads"
echo "Monitor: tail -f /mnt/volume/logs/*.log"
echo "Check space: df -h /mnt/volume"
