#!/bin/bash
# Fix CUDA symlink and flashinfer cache on all Qwen3.5 machines
KEY="$HOME/.ssh/rccg_key"
OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"

# Get IPs from inventory
IPS="62.169.159.176 185.216.20.70 185.216.21.62 38.128.232.129 38.128.232.131 69.19.137.100"
NAMES="fada-1 fada-2 fada-3 fada-4 fada-6 fada-8"

i=0
for ip in $IPS; do
  name=$(echo $NAMES | cut -d' ' -f$((i+1)))
  echo "=== Fixing $name ($ip) ==="
  ssh -i "$KEY" $OPTS "ubuntu@$ip" 'sudo rm -f /usr/local/cuda && sudo ln -s /usr/local/cuda-12.8 /usr/local/cuda && rm -rf ~/.cache/flashinfer && rm -f /home/ubuntu/fada-results/checkpoint_*.json /home/ubuntu/fada-results/eval_done.marker && echo "CUDA:" && /usr/local/cuda/bin/nvcc --version 2>&1 | tail -1 && echo "FIXED"' &
  i=$((i+1))
done

wait
echo "=== ALL DONE ==="
