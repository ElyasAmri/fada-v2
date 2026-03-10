#!/usr/bin/env bash
set -euo pipefail

export PIP_CACHE_DIR=/mnt/models/pip-cache

echo "=== Installing CUDA toolkit and flash-attn ==="

# Install CUDA toolkit (needed to compile flash-attn)
echo "--- Installing CUDA toolkit ---"
if ! command -v nvcc &>/dev/null; then
    # Install just the CUDA toolkit (not driver) via conda in ft-unsloth
    eval "$(/home/ubuntu/miniconda3/bin/conda shell.bash hook)"
    conda activate ft-unsloth
    conda install -y -c nvidia cuda-toolkit=12.4 -q 2>&1 | tail -5

    echo "nvcc version:"
    nvcc --version 2>/dev/null || echo "nvcc still not found"

    echo "--- Installing flash-attn ---"
    export CUDA_HOME=$CONDA_PREFIX
    pip install flash-attn --no-build-isolation 2>&1 | tail -10
    python -c "import flash_attn; print('flash-attn:', flash_attn.__version__)" 2>/dev/null || echo "flash-attn: FAILED"
    conda deactivate
else
    echo "CUDA toolkit already available"
fi

echo "=== Done ==="
