#!/bin/bash
# Thin wrapper to call rccg.sh via WSL with clean PATH (no Windows PATH pollution)
# Usage: ./r.sh status | ./r.sh play setup --limit fada-3 | ./r.sh ssh fada-3 "cmd"
RCCG="/mnt/c/Users/elyas/workspace/fada-v3/experiments/rccg/rccg.sh"
wsl -e bash --norc --noprofile -c "export PATH=\"\$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin\"; $RCCG $*"
