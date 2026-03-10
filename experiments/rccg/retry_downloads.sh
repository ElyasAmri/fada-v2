#!/bin/bash
HF="$HOME/.local/bin/hf"

# Expected minimum sizes (bytes) - rough estimates
# If downloaded size is less, re-download
declare -A MODELS
MODELS[/mnt/volume]="Qwen/Qwen2.5-VL-3B-Instruct"
MODELS[/mnt/vol0]="OpenGVLab/InternVL3_5-8B"
MODELS[/mnt/vol1]="llava-hf/llava-onevision-qwen2-7b-ov-hf"
MODELS[/mnt/vol2]="microsoft/Phi-4-multimodal-instruct"
MODELS[/mnt/vol3]="deepseek-ai/deepseek-vl2-small"
MODELS[/mnt/vol4]="openbmb/MiniCPM-V-4_5"
MODELS[/mnt/vol5]="google/gemma-3-12b-it"

for MOUNT in /mnt/volume /mnt/vol0 /mnt/vol1 /mnt/vol2 /mnt/vol3 /mnt/vol4 /mnt/vol5; do
  MODEL="${MODELS[$MOUNT]}"
  USED_KB=$(df "$MOUNT" --output=used | tail -1 | tr -d ' ')
  USED_MB=$((USED_KB / 1024))
  echo ""
  echo "=== $MOUNT: $MODEL (${USED_MB}MB) ==="

  if [ "$USED_MB" -lt 5000 ]; then
    echo "  Incomplete (${USED_MB}MB). Re-downloading sequentially..."
    HF_HOME=${MOUNT}/huggingface $HF download "$MODEL" 2>&1 | tail -3
    echo "  Done."
  else
    echo "  Looks complete (${USED_MB}MB). Skipping."
  fi
done

echo ""
echo "=== FINAL DISK USAGE ==="
df -h /mnt/volume /mnt/vol0 /mnt/vol1 /mnt/vol2 /mnt/vol3 /mnt/vol4 /mnt/vol5
