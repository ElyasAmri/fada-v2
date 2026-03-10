#!/usr/bin/env bash
set -euo pipefail

# Run the framework comparison training queue on RCCG
# All outputs go to the volume to save root disk space

export HF_HOME=/mnt/models/huggingface
export PIP_CACHE_DIR=/mnt/models/pip-cache

PROJECT_DIR=/home/ubuntu/fada-v3
VENV=$PROJECT_DIR/venv
OUTPUT_BASE=/mnt/models/fc-runs

cd "$PROJECT_DIR"

echo "=== Framework Comparison Training ==="
echo "Output dir: $OUTPUT_BASE"
echo "HF_HOME: $HF_HOME"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
df -h / /mnt/models

$VENV/bin/python experiments/framework_comparison/run_queue.py \
    --output-base "$OUTPUT_BASE" \
    "$@"
