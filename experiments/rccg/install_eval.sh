#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/mnt/models/huggingface

eval "$(/home/ubuntu/miniconda3/bin/conda shell.bash hook)"
conda activate ft-eval

echo "Installing eval packages in ft-eval..."
pip install vllm scikit-learn sentence-transformers openpyxl bert-score python-dotenv num2words -q 2>&1 | tail -5

echo "Verifying..."
python -c "import vllm; print('vllm:', vllm.__version__)"
python -c "import sklearn; print('sklearn: OK')"
python -c "import sentence_transformers; print('sentence_transformers: OK')"

conda deactivate
echo "ft-eval setup complete"
