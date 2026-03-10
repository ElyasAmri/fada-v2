#!/usr/bin/env bash
set -euo pipefail
eval "$(/home/ubuntu/miniconda3/bin/conda shell.bash hook)"
FC_DIR=/home/ubuntu/fada-v3/experiments/framework_comparison

install_env() {
    local name=$1
    local req=$2
    echo "=== $name ==="
    conda activate "$name"
    if ! python -c 'import torch' 2>/dev/null; then
        echo "Installing PyTorch..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 -q 2>&1 | tail -1
    else
        echo "PyTorch already installed"
    fi
    echo "Installing requirements..."
    pip install -r "$FC_DIR/setup/$req" -q 2>&1 | tail -3
    conda deactivate
    echo "$name: done"
}

install_env ft-unsloth requirements_unsloth.txt
install_env ft-llamafactory requirements_llamafactory.txt
install_env ft-axolotl requirements_axolotl.txt

# Install flash-attn from pre-built wheel for axolotl
echo "=== Installing flash-attn for ft-axolotl ==="
conda activate ft-axolotl
pip install flash-attn --no-build-isolation -q 2>&1 | tail -3 || echo "flash-attn install failed (non-fatal)"
conda deactivate

# Also install eval packages in the venv
echo "=== Installing eval packages ==="
/home/ubuntu/fada-v3/venv/bin/pip install vllm scikit-learn sentence-transformers openpyxl bert-score python-dotenv num2words -q 2>&1 | tail -3

echo "=== All envs done ==="
