#!/bin/bash
# Shortcut to rccg.sh from project root
# Usage: ./r.sh ssh fada-1 hostname
#        ./r.sh status
#        ./r.sh play setup --limit fada-1
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")/experiments/rccg"

if [ "${1:-}" = "play" ]; then
  # Ansible requires WSL -- convert Git Bash path to WSL path
  RCCG_PATH="$(cd "$SCRIPT_DIR" && pwd)/rccg.sh"
  WSL_RCCG=$(echo "$RCCG_PATH" | sed 's|^/\([a-z]\)/|/mnt/\1/|')
  WSL_DIR=$(echo "$(cd "$SCRIPT_DIR" && pwd)" | sed 's|^/\([a-z]\)/|/mnt/\1/|')
  wsl -e bash --norc --noprofile -c "
    export PATH=\"\$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin\"
    export RCCG_SCRIPT_DIR='$WSL_DIR'
    tr -d '\r' < '$WSL_RCCG' > /tmp/rccg_run.sh
    bash /tmp/rccg_run.sh $*
  "
else
  exec "$SCRIPT_DIR/rccg.sh" "$@"
fi
