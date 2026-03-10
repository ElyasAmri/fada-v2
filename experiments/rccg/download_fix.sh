#!/bin/bash
HF="$HOME/.local/bin/hf"
TOKEN="$1"
export HF_HOME=/mnt/volume/huggingface

MODELS=(
  "Qwen/Qwen3-VL-4B-Instruct"
  "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
  "Qwen/Qwen3-VL-8B-Thinking"
)

for MODEL in "${MODELS[@]}"; do
  echo "=== $MODEL ==="
  HF_TOKEN="$TOKEN" $HF download "$MODEL" 2>&1 | tail -2
  echo ""
done

echo "=== DONE ==="
du -sh $HF_HOME/hub/
df -h /mnt/volume | tail -1
