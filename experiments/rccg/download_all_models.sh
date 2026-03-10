#!/bin/bash
HF="$HOME/.local/bin/hf"
TOKEN="$1"

# Use the first volume as the master
export HF_HOME=/mnt/volume/huggingface

MODELS=(
  # Already downloaded (will be skipped by hf download)
  "Qwen/Qwen2.5-VL-3B-Instruct"
  "OpenGVLab/InternVL3_5-8B"
  "llava-hf/llava-onevision-qwen2-7b-ov-hf"
  "microsoft/Phi-4-multimodal-instruct"
  "deepseek-ai/deepseek-vl2-small"
  "openbmb/MiniCPM-V-4_5"
  "google/gemma-3-12b-it"
  "google/medgemma-4b-it"
  # New models to download
  "Qwen/Qwen3-VL-8B-Instruct"
  "Qwen/Qwen3-VL-4B"
  "Qwen/Qwen3-VL-2B"
  "OpenGVLab/InternVL3_5-4B"
  "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
  "google/gemma-3-4b-it"
  "google/gemma-3-27b-it"
  "HuggingFaceTB/SmolVLM-2.2B-Instruct"
  "microsoft/llava-med-v1.5-mistral-7b"
  "Qwen/Qwen3-VL-8B-Instruct-Thinking"
)

# Gated models need token
GATED_PREFIXES="google/ mistralai/"

echo "Downloading ${#MODELS[@]} models to $HF_HOME"
echo "================================================"

for MODEL in "${MODELS[@]}"; do
  echo ""
  echo "=== $MODEL ==="

  # Check if already downloaded
  SAFE=$(echo "$MODEL" | tr "/" "--")
  if [ -d "$HF_HOME/hub/models--${SAFE}/blobs" ]; then
    BSIZE=$(du -sh "$HF_HOME/hub/models--${SAFE}" 2>/dev/null | cut -f1)
    echo "  Already cached (${BSIZE}). Verifying..."
  fi

  # Use token for gated models
  NEEDS_TOKEN=false
  for PREFIX in $GATED_PREFIXES; do
    case "$MODEL" in $PREFIX*) NEEDS_TOKEN=true ;; esac
  done

  if $NEEDS_TOKEN && [ -n "$TOKEN" ]; then
    HF_TOKEN="$TOKEN" $HF download "$MODEL" 2>&1 | tail -2
  else
    $HF download "$MODEL" 2>&1 | tail -2
  fi
done

echo ""
echo "================================================"
echo "=== DONE ==="
du -sh $HF_HOME/hub/
df -h /mnt/volume
