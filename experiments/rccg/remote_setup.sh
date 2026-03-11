#!/bin/bash
# Remote setup script for framework comparison on RCCG H100
# Upload and run: scp this.sh host:~ && ssh host 'bash ~/remote_setup.sh'
set -euo pipefail

echo "=== RCCG Framework Comparison Setup ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

PROJECT_DIR="/home/ubuntu/fada-v3"
CONDA_DIR="/home/ubuntu/miniconda3"

# 1. Update repo
echo "--- Updating repo ---"
cd "$PROJECT_DIR"
git pull

# 2. Install miniconda if missing
if [ ! -f "$CONDA_DIR/bin/conda" ]; then
    echo "--- Installing miniconda ---"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O /home/ubuntu/miniconda_install.sh
    bash /home/ubuntu/miniconda_install.sh -b -p "$CONDA_DIR"
    rm -f /home/ubuntu/miniconda_install.sh
    echo "Conda installed: $($CONDA_DIR/bin/conda --version)"
else
    echo "Conda already installed: $($CONDA_DIR/bin/conda --version)"
fi

# Init conda for this shell
eval "$($CONDA_DIR/bin/conda shell.bash hook)"

# 3. Create venv for eval using conda (avoids system python venv issues)
VENV="$PROJECT_DIR/venv"
if [ ! -f "$VENV/bin/pip" ]; then
    echo "--- Creating eval venv via conda ---"
    rm -rf "$VENV"
    conda create -p "$VENV" python=3.10 pip -y -q
fi

echo "--- Installing eval packages ---"
"$VENV/bin/pip" install --upgrade pip -q
"$VENV/bin/pip" install vllm scikit-learn sentence-transformers openpyxl bert-score python-dotenv num2words -q 2>&1 | tail -3

# 4. Check dataset
echo "--- Checking dataset ---"
if [ -d "$PROJECT_DIR/data/Fetal Ultrasound" ]; then
    COUNT=$(find "$PROJECT_DIR/data/Fetal Ultrasound" -name "*.png" | wc -l)
    echo "$COUNT images found"
else
    echo "WARNING: Dataset not found at $PROJECT_DIR/data/Fetal Ultrasound"
    echo "Upload dataset before running training."
fi

# 5. Check training data
if [ -f "$PROJECT_DIR/data/vlm_training/gt_train.jsonl" ]; then
    TRAIN_COUNT=$(wc -l < "$PROJECT_DIR/data/vlm_training/gt_train.jsonl")
    echo "$TRAIN_COUNT training samples"
else
    echo "WARNING: Training data not found"
fi

# 6. Create conda environments
echo "--- Setting up conda environments ---"
FC_DIR="$PROJECT_DIR/experiments/framework_comparison"

setup_env() {
    local name="$1"
    local req="$2"
    if conda env list | grep -q "^${name} "; then
        echo "$name: already exists"
        return
    fi
    echo "$name: creating..."
    conda create -n "$name" python=3.10 -y -q
    conda activate "$name"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 -q 2>&1 | tail -1
    pip install -r "$FC_DIR/setup/$req" -q 2>&1 | tail -1
    conda deactivate
    echo "$name: done"
}

setup_env "ft-unsloth" "requirements_unsloth.txt"
setup_env "ft-llamafactory" "requirements_llamafactory.txt"
setup_env "ft-swift" "requirements_swift.txt"
setup_env "ft-axolotl" "requirements_axolotl.txt"
setup_env "ft-eval" "requirements_eval.txt"

# 7. Convert data to ShareGPT format
echo "--- Data conversion ---"
if [ ! -f "$PROJECT_DIR/data/vlm_training/gt_train_sharegpt.jsonl" ]; then
    "$VENV/bin/python" "$FC_DIR/convert_to_sharegpt.py" \
        --input "$PROJECT_DIR/data/vlm_training/gt_train.jsonl"
    "$VENV/bin/python" "$FC_DIR/convert_to_sharegpt.py" \
        --input "$PROJECT_DIR/data/vlm_training/gt_val.jsonl"
else
    echo "ShareGPT data already exists"
fi

# 8. Dry run
echo "--- Run matrix ---"
"$VENV/bin/python" "$FC_DIR/run_queue.py" --dry-run

echo ""
echo "=== Setup complete ==="
echo "To start training: $VENV/bin/python $FC_DIR/run_queue.py"
