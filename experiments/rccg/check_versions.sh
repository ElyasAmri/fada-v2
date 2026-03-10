#!/usr/bin/env bash
set -euo pipefail
eval "$(/home/ubuntu/miniconda3/bin/conda shell.bash hook)"

for env in ft-unsloth ft-llamafactory ft-swift ft-axolotl; do
    echo "=== $env ==="
    conda activate "$env"
    python -c "import torch; print('torch:', torch.__version__)" 2>/dev/null || echo "torch: NOT FOUND"
    python -c "import unsloth; print('unsloth:', unsloth.__version__)" 2>/dev/null || true
    python -c "import llamafactory; print('llamafactory:', llamafactory.__version__)" 2>/dev/null || true
    conda deactivate
done
