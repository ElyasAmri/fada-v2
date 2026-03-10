#!/usr/bin/env bash
set -euo pipefail

# Don't use cross-device pip cache
unset PIP_CACHE_DIR

echo "=== Installing flash-attn ==="
eval "$(/home/ubuntu/miniconda3/bin/conda shell.bash hook)"
conda activate ft-unsloth

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
echo "nvcc: $(nvcc --version 2>&1 | grep release)"

echo "--- Building flash-attn (this takes ~10 min) ---"
pip install flash-attn --no-build-isolation 2>&1 | tail -15
python -c "import flash_attn; print('flash-attn:', flash_attn.__version__)" 2>/dev/null || echo "flash-attn: FAILED"

conda deactivate
echo "=== Done ==="
