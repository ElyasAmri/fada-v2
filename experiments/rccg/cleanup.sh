#!/bin/bash
# Clean old pip --user installs on machines that need venvs
for h in fada-2 fada-4 fada-5 fada-6 fada-7 fada-8; do
  ssh -o ConnectTimeout=10 "$h" 'rm -rf ~/.cache/pip ~/.local/lib/python*/site-packages 2>/dev/null; echo cleaned' &
done
wait
echo "All done"
