#!/bin/bash
# Convenience wrapper: runs ansible-playbook via WSL
# Usage:
#   ./run.sh status                              # Run status playbook
#   ./run.sh exec -e "cmd='nvidia-smi'"          # Run arbitrary command
#   ./run.sh setup --limit fada-1                # Setup specific machine
#   ./run.sh run_eval                            # Launch sharded eval
#   ./run.sh collect                             # Fetch results
#   ./run.sh stop                                # Stop eval on all machines
#   ./run.sh ping                                # Test connectivity

PLAYBOOK="$1"
shift

if [ -z "$PLAYBOOK" ]; then
    echo "Usage: ./run.sh <playbook> [ansible-playbook args...]"
    echo ""
    echo "Available playbooks:"
    echo "  ping       - Test connectivity"
    echo "  status     - Check vLLM + eval progress"
    echo "  setup      - Full machine setup (idempotent)"
    echo "  run_eval   - Launch sharded evaluation"
    echo "  collect    - Fetch checkpoints and results"
    echo "  stop       - Stop eval on all machines"
    echo "  exec       - Run command: ./run.sh exec -e \"cmd='...'\""
    exit 1
fi

WSL_PROJECT="/mnt/c/Users/elyas/workspace/fada-v3/experiments/rccg"

if [ "$PLAYBOOK" = "ping" ]; then
    wsl -e bash -c "export PATH=\$HOME/.local/bin:\$PATH && cd $WSL_PROJECT && ANSIBLE_CONFIG=./ansible.cfg ansible all -m ping"
else
    wsl -e bash -c "export PATH=\$HOME/.local/bin:\$PATH && cd $WSL_PROJECT && ANSIBLE_CONFIG=./ansible.cfg ansible-playbook playbooks/${PLAYBOOK}.yml $*"
fi
