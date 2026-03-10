#!/usr/bin/env bash
echo "Killing all training processes..."
pkill -9 -f "train_unsloth" 2>/dev/null || true
pkill -9 -f "train_llamafactory" 2>/dev/null || true
pkill -9 -f "train_swift" 2>/dev/null || true
pkill -9 -f "train_axolotl" 2>/dev/null || true
pkill -9 -f "run_queue" 2>/dev/null || true
pkill -9 -f "run_training" 2>/dev/null || true
sleep 3
echo "Remaining python:"
ps aux | grep python | grep -v grep || echo "none"
echo "GPU memory:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi failed"
