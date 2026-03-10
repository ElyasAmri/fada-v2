#!/bin/bash
# Upgrade vLLM nightly + transformers on specified machines
KEY="$HOME/.ssh/rccg_key"
OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"
CMD='source /home/ubuntu/fada-v3/venv/bin/activate && pip install vllm==0.16.1rc1.dev268 --extra-index-url https://wheels.vllm.ai/nightly --pre 2>&1 | tail -3 && pip install --upgrade transformers 2>&1 | tail -3 && echo "VERSIONS:" && pip show vllm | grep Version && pip show transformers | grep Version'

IPS="185.216.20.70 185.216.21.62 38.128.232.129 38.128.232.131 69.19.137.100"
NAMES="fada-2 fada-3 fada-4 fada-6 fada-8"

i=0
for ip in $IPS; do
  name=$(echo $NAMES | cut -d' ' -f$((i+1)))
  echo "=== Upgrading $name ($ip) ==="
  ssh -i "$KEY" $OPTS "ubuntu@$ip" "bash -c '$CMD'" &
  i=$((i+1))
done

wait
echo "=== ALL DONE ==="
