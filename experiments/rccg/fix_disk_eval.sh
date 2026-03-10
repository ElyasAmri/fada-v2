#!/usr/bin/env bash
set -euo pipefail

echo "=== Fix disk and install ft-eval ==="

# 1. Clean conda caches
echo "Cleaning conda caches..."
/home/ubuntu/miniconda3/bin/conda clean --all -y 2>&1 | tail -3

# Clean pip caches in all envs
rm -rf /home/ubuntu/.cache/pip
rm -rf /tmp/pip-*

echo "Root disk after cleanup:"
df -h /

# 2. Remove ft-eval from root (if exists) and recreate on volume
eval "$(/home/ubuntu/miniconda3/bin/conda shell.bash hook)"

if conda env list | grep -q "^ft-eval "; then
    echo "Removing ft-eval from root..."
    conda env remove -n ft-eval -y 2>&1 | tail -3
fi

# Create ft-eval on the volume
EVAL_PREFIX=/mnt/models/conda-envs/ft-eval
if [ -d "$EVAL_PREFIX" ]; then
    echo "ft-eval already exists on volume"
else
    echo "Creating ft-eval on volume..."
    mkdir -p /mnt/models/conda-envs
    conda create -p "$EVAL_PREFIX" python=3.10 -y -q
fi

echo "Installing eval packages..."
conda activate "$EVAL_PREFIX"
export PIP_CACHE_DIR=/mnt/models/pip-cache
mkdir -p "$PIP_CACHE_DIR"
pip install vllm scikit-learn sentence-transformers openpyxl bert-score python-dotenv num2words -q 2>&1 | tail -5
echo "Verifying..."
python -c "import vllm; print('vllm:', vllm.__version__)"
python -c "import sklearn; print('sklearn: OK')"
conda deactivate

echo "Root disk after install:"
df -h /
echo "Volume after install:"
df -h /mnt/models

echo "=== Done ==="
echo "NOTE: ft-eval is at $EVAL_PREFIX (not a named env)"
