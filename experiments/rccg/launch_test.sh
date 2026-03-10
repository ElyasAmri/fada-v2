#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/fada-v3
nohup /usr/bin/bash /home/ubuntu/run_training.sh --test-run --filter-model qwen2.5-vl-7b --filter-framework unsloth --skip-eval > /home/ubuntu/training_test.log 2>&1 &
echo "Launched PID: $!"
sleep 2
head -5 /home/ubuntu/training_test.log 2>/dev/null || echo "Log not ready yet"
