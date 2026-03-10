#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/home/ubuntu/fada-v3
VENV=$PROJECT_DIR/venv
DATA_DIR=/mnt/models/dataset

if [ -d "$DATA_DIR/Fetal Ultrasound" ]; then
    echo "Dataset already exists"
    find "$DATA_DIR/Fetal Ultrasound" -name "*.png" | wc -l
    exit 0
fi

echo "Installing gdown..."
$VENV/bin/pip install gdown -q

echo "Downloading dataset from Google Drive..."
$VENV/bin/gdown --id 15TFlu6NXSBbrskfyecdU7IHG6DyeJXpQ -O /tmp/fetal_ultrasound.zip

SIZE=$(stat -c%s /tmp/fetal_ultrasound.zip 2>/dev/null || echo 0)
echo "Downloaded $(( SIZE / 1048576 ))MB"

if [ "$SIZE" -lt 1000000000 ]; then
    echo "FATAL: Download too small (${SIZE} bytes)"
    rm -f /tmp/fetal_ultrasound.zip
    exit 1
fi

echo "Extracting..."
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"
unzip -q /tmp/fetal_ultrasound.zip
rm -f /tmp/fetal_ultrasound.zip

# Symlink into project data dir
ln -sfn "$DATA_DIR/Fetal Ultrasound" "$PROJECT_DIR/data/Fetal Ultrasound"

COUNT=$(find "$DATA_DIR/Fetal Ultrasound" -name "*.png" | wc -l)
echo "Done: $COUNT images"
