#!/usr/bin/env bash
set -euo pipefail

export PIP_CACHE_DIR=/mnt/models/pip-cache

echo "=== Installing flash-attn ==="

eval "$(/home/ubuntu/miniconda3/bin/conda shell.bash hook)"

# Install in ft-unsloth
echo "--- ft-unsloth ---"
conda activate ft-unsloth
pip install flash-attn --no-build-isolation -q 2>&1 | tail -5
python -c "import flash_attn; print('flash-attn:', flash_attn.__version__)" 2>/dev/null || echo "flash-attn: FAILED"
conda deactivate

# Install in ft-axolotl (also needs it)
echo "--- ft-axolotl ---"
conda activate ft-axolotl
pip install flash-attn --no-build-isolation -q 2>&1 | tail -5
python -c "import flash_attn; print('flash-attn:', flash_attn.__version__)" 2>/dev/null || echo "flash-attn: FAILED"
conda deactivate

echo "=== Done ==="
