#!/usr/bin/env bash
set -euo pipefail
eval "$(/home/ubuntu/miniconda3/bin/conda shell.bash hook)"
conda activate ft-unsloth
python /tmp/test_fa.py
conda deactivate
