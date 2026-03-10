#!/usr/bin/env bash
set -euo pipefail
export PIP_CACHE_DIR=/mnt/models/pip-cache
eval "$(/home/ubuntu/miniconda3/bin/conda shell.bash hook)"

echo "=== Fixing ft-unsloth PyTorch ==="
conda activate ft-unsloth
echo "Current torch: $(python -c 'import torch; print(torch.__version__)')"
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu124 -q 2>&1 | tail -5
echo "New torch: $(python -c 'import torch; print(torch.__version__)')"

echo "Testing unsloth import..."
python -c "from unsloth import FastModel; print('Unsloth OK')" 2>&1 | tail -3
conda deactivate

echo "=== Fixing ft-axolotl PyTorch (cu128 -> cu124) ==="
conda activate ft-axolotl
echo "Current torch: $(python -c 'import torch; print(torch.__version__)')"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 -q 2>&1 | tail -5
echo "New torch: $(python -c 'import torch; print(torch.__version__)')"
conda deactivate

echo "=== Done ==="
