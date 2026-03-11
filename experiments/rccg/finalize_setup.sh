#!/usr/bin/env bash
set -euo pipefail

echo "=== Finalize RCCG Setup ==="

# 1. Set HF_HOME to volume
if ! grep -q HF_HOME /home/ubuntu/.bashrc 2>/dev/null; then
    echo 'export HF_HOME=/mnt/models/huggingface' >> /home/ubuntu/.bashrc
    echo "HF_HOME added to .bashrc"
else
    echo "HF_HOME already in .bashrc"
fi
mkdir -p /mnt/models/huggingface
echo "HF_HOME=/mnt/models/huggingface"

# 2. Convert training data to ShareGPT format
PROJECT_DIR=/home/ubuntu/fada-v3
VENV=$PROJECT_DIR/venv
cd "$PROJECT_DIR"

if [ -f "data/vlm_training/gt_train_sharegpt.jsonl" ]; then
    echo "ShareGPT train data already exists"
else
    echo "Converting train data to ShareGPT..."
    $VENV/bin/python experiments/framework_comparison/convert_to_sharegpt.py \
        --input data/vlm_training/gt_train.jsonl
    echo "Train conversion done"
fi

if [ -f "data/vlm_training/gt_val_sharegpt.jsonl" ]; then
    echo "ShareGPT val data already exists"
else
    echo "Converting val data to ShareGPT..."
    $VENV/bin/python experiments/framework_comparison/convert_to_sharegpt.py \
        --input data/vlm_training/gt_val.jsonl
    echo "Val conversion done"
fi

# 3. Create ft-eval conda env if missing
eval "$(/home/ubuntu/miniconda3/bin/conda shell.bash hook)"
if conda env list | grep -q "^ft-eval "; then
    echo "ft-eval env already exists"
else
    echo "Creating ft-eval env..."
    conda create -n ft-eval python=3.10 -y -q
    conda activate ft-eval
    pip install vllm scikit-learn sentence-transformers openpyxl bert-score python-dotenv num2words -q 2>&1 | tail -3
    conda deactivate
    echo "ft-eval env created"
fi

# 4. Dry run
echo "=== Dry Run ==="
export HF_HOME=/mnt/models/huggingface
$VENV/bin/python experiments/framework_comparison/run_queue.py --dry-run

echo "=== Finalize complete ==="
