#!/usr/bin/env bash
set -euo pipefail

# Clean up test run
rm -rf /mnt/models/fc-runs/qwen2.5-vl-7b_unsloth /mnt/models/fc-runs/runs.jsonl

# Launch full training queue in background
cd /home/ubuntu/fada-v3
nohup /usr/bin/bash /home/ubuntu/run_training.sh > /home/ubuntu/training_full.log 2>&1 &
echo "Launched full training queue PID: $!"
echo "Monitor with: tail -f /home/ubuntu/training_full.log"
