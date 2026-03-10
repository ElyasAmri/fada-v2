#!/usr/bin/env bash
set -euo pipefail

HOST=69.19.137.143
KEY=~/.ssh/rccg_key
SCP="scp -i $KEY -o StrictHostKeyChecking=no"
SSH="ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$HOST"

PROJECT=/home/ubuntu/fada-v3
WIN_PROJECT=/mnt/c/Users/elyas/workspace/fada-v3

echo "=== Uploading scoring files ==="
$SCP "$WIN_PROJECT/data/Fetal Ultrasound Annotations Normalized.xlsx" "ubuntu@$HOST:$PROJECT/data/"
$SCP "$WIN_PROJECT/data/dataset_splits.json" "ubuntu@$HOST:$PROJECT/data/"
echo "Scoring files uploaded"

echo "=== Uploading finalize script ==="
$SCP "$WIN_PROJECT/experiments/rccg/finalize_setup.sh" "ubuntu@$HOST:/home/ubuntu/"
$SSH "sed -i 's/\r$//' /home/ubuntu/finalize_setup.sh"

echo "=== Running finalize ==="
$SSH "/usr/bin/bash /home/ubuntu/finalize_setup.sh"
