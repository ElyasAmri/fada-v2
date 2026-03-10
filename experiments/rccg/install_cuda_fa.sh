#!/usr/bin/env bash
set -euo pipefail

export PIP_CACHE_DIR=/mnt/models/pip-cache

echo "=== Installing CUDA toolkit via conda + flash-attn ==="

eval "$(/home/ubuntu/miniconda3/bin/conda shell.bash hook)"
conda activate ft-unsloth

# Install minimal CUDA dev packages via conda (just nvcc + headers)
echo "--- Installing CUDA nvcc + headers in conda env ---"
conda install -y -c nvidia/label/cuda-12.4.0 cuda-nvcc cuda-cudart-dev cuda-libraries-dev -q 2>&1 | tail -5

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
echo "nvcc: $(nvcc --version 2>&1 | grep release || echo 'not found')"

echo "--- Installing flash-attn ---"
pip install flash-attn --no-build-isolation 2>&1 | tail -10
python -c "import flash_attn; print('flash-attn:', flash_attn.__version__)" 2>/dev/null || echo "flash-attn: FAILED"

echo "--- Testing with Unsloth ---"
python -c "
from unsloth import FastModel
import torch
model, proc = FastModel.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct', max_seq_length=4096, load_in_4bit=True, dtype=torch.bfloat16)
print('Model loaded successfully')
" 2>&1 | grep -E "FA|Flash|flash|Loaded|Error" | head -5

conda deactivate
echo "=== Done ==="
