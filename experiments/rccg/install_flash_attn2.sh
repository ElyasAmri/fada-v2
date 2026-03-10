#!/usr/bin/env bash
set -euo pipefail

export PIP_CACHE_DIR=/mnt/models/pip-cache

echo "=== Installing flash-attn from pre-built wheel ==="

eval "$(/home/ubuntu/miniconda3/bin/conda shell.bash hook)"

# ft-unsloth: torch 2.6, cuda 12.4, python 3.10
echo "--- ft-unsloth ---"
conda activate ft-unsloth
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_VER=$(python -c "import torch; print(torch.version.cuda.replace('.',''))")
PY_VER=$(python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
echo "torch=$TORCH_VER cuda=$CUDA_VER python=$PY_VER"

# Install from flashattn.dev pre-built wheels
pip install flash-attn --no-build-isolation --prefer-binary 2>&1 | tail -10
# If that fails, try the index URL
if ! python -c "import flash_attn" 2>/dev/null; then
    echo "Trying alternative install..."
    pip install flash-attn -f "https://github.com/Dao-AILab/flash-attention/releases" --no-build-isolation 2>&1 | tail -10
fi
python -c "import flash_attn; print('flash-attn:', flash_attn.__version__)" 2>/dev/null || echo "flash-attn: FAILED"
conda deactivate

echo "=== Done ==="
