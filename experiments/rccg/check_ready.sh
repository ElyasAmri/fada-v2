#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/home/ubuntu/fada-v3
VENV=$PROJECT_DIR/venv

echo "=== Readiness Check ==="

# Check ft-eval packages
eval "$(/home/ubuntu/miniconda3/bin/conda shell.bash hook)"
conda activate ft-eval
python -c "import vllm; print('vllm:', vllm.__version__)" 2>/dev/null || echo "vllm: NOT installed"
conda deactivate

# Check HF_HOME
source /home/ubuntu/.bashrc 2>/dev/null || true
echo "HF_HOME=${HF_HOME:-NOT SET}"

# Check data files
echo "--- Data ---"
ls -lh "$PROJECT_DIR/data/vlm_training/"*sharegpt* 2>/dev/null || echo "ShareGPT files: MISSING"
ls -lh "$PROJECT_DIR/data/dataset_splits.json" 2>/dev/null || echo "dataset_splits.json: MISSING"
ls "$PROJECT_DIR/data/"*.xlsx 2>/dev/null || echo "Annotations: MISSING"

# Dry run
echo "--- Dry Run ---"
export HF_HOME=/mnt/models/huggingface
cd "$PROJECT_DIR"
$VENV/bin/python experiments/framework_comparison/run_queue.py --dry-run
