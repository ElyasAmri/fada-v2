#!/bin/bash
# RCCG cluster management CLI
# Usage: ./rccg.sh <command> [args]
#   play <playbook> [--limit hosts]   Run ansible-playbook
#   ssh <host> [cmd]                  SSH into a machine (or run command)
#   status                            Check eval progress on all machines
#   logs <host>                       Tail eval log on a machine

set -euo pipefail

# Fix PATH on WSL to avoid Windows path pollution (spaces/parens break bash)
if [ -f /proc/version ] && grep -q Microsoft /proc/version 2>/dev/null; then
  clean_path=""
  IFS=':' read -ra PARTS <<< "$PATH"
  for p in "${PARTS[@]}"; do
    case "$p" in /mnt/c/*|/mnt/d/*) ;; *) clean_path="${clean_path:+$clean_path:}$p" ;; esac
  done
  export PATH="$clean_path"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INVENTORY="$SCRIPT_DIR/inventory/hosts.yml"
SSH_KEY="$HOME/.ssh/rccg_key"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o LogLevel=ERROR"

# Parse host IP from inventory YAML (simple grep, no yq needed)
get_ip() {
  local host="$1"
  local ip
  ip=$(grep -A5 "^\s*${host}:" "$INVENTORY" | grep ansible_host | awk '{print $NF}' | head -1)
  if [ -z "$ip" ]; then
    echo "Error: could not resolve IP for host '$host'" >&2
    exit 1
  fi
  echo "$ip"
}

# Get all host names
get_hosts() {
  grep -oP '^\s{8}fada-\d+' "$INVENTORY" | awk '{print $1}'
}

# Get model for a host
get_model() {
  local host="$1"
  local model
  model=$(grep -A5 "^\s*${host}:" "$INVENTORY" | grep vllm_model | awk '{print $NF}' | head -1)
  if [ -z "$model" ]; then
    echo "Error: could not resolve model for host '$host'" >&2
    exit 1
  fi
  echo "$model"
}

cmd_play() {
  local playbook="$1"; shift
  # Accept bare name or full path
  if [[ "$playbook" != */* ]]; then
    playbook="$SCRIPT_DIR/playbooks/${playbook}.yml"
  fi

  # Detect environment
  if grep -q Microsoft /proc/version 2>/dev/null || [ -d /mnt/c ]; then
    # Already in WSL
    local tmp_cfg="/tmp/rccg_ansible.cfg"
    local tmp_inv="/tmp/rccg_hosts.yml"
    cp "$SCRIPT_DIR/ansible.cfg" "$tmp_cfg"
    cp "$INVENTORY" "$tmp_inv"
    chmod 644 "$tmp_cfg" "$tmp_inv"
    ANSIBLE_CONFIG="$tmp_cfg" ansible-playbook -i "$tmp_inv" "$playbook" "$@"
  else
    # Git Bash on Windows - delegate to WSL
    local wsl_script="$SCRIPT_DIR/ansible.cfg"
    local wsl_inv="$INVENTORY"
    local wsl_pb="$playbook"
    # Convert Windows paths to WSL paths
    wsl_script=$(wslpath -u "$wsl_script" 2>/dev/null || echo "$wsl_script")
    wsl_inv=$(wslpath -u "$wsl_inv" 2>/dev/null || echo "$wsl_inv")
    wsl_pb=$(wslpath -u "$wsl_pb" 2>/dev/null || echo "$wsl_pb")
    wsl -e bash -lc "cp '$wsl_script' /tmp/rccg_ansible.cfg && cp '$wsl_inv' /tmp/rccg_hosts.yml && chmod 644 /tmp/rccg_ansible.cfg /tmp/rccg_hosts.yml && ANSIBLE_CONFIG=/tmp/rccg_ansible.cfg ansible-playbook -i /tmp/rccg_hosts.yml '$wsl_pb' $*"
  fi
}

cmd_ssh() {
  local host="$1"; shift
  local ip=$(get_ip "$host")
  if [ -z "$ip" ]; then
    echo "Unknown host: $host" >&2; exit 1
  fi
  if [ $# -eq 0 ]; then
    ssh -i "$SSH_KEY" $SSH_OPTS "ubuntu@$ip"
  else
    ssh -i "$SSH_KEY" $SSH_OPTS "ubuntu@$ip" "$@"
  fi
}

cmd_status() {
  printf "%-8s %-45s %s\n" "HOST" "MODEL" "STATUS"
  printf "%-8s %-45s %s\n" "--------" "---------------------------------------------" "----------"
  for host in $(get_hosts); do
    local ip=$(get_ip "$host")
    local model=$(get_model "$host")
    local short_model=$(basename "$model")
    local result
    result=$(ssh -i "$SSH_KEY" $SSH_OPTS "ubuntu@$ip" '
      if [ -f /home/ubuntu/fada-results/eval_done.marker ]; then
        code=$(grep exit_code /home/ubuntu/fada-results/eval_done.marker | cut -d= -f2)
        if [ "$code" = "0" ]; then echo "DONE"; else echo "FAILED(rc=$code)"; fi
      elif pgrep -f "[t]est_api_vlm" > /dev/null; then
        prog=$(tail -1 /home/ubuntu/fada-results/eval_run.log 2>/dev/null | grep -oP "\d+/\d+" | tail -1)
        echo "RUNNING ${prog:-?/?}"
      else
        echo "IDLE"
      fi
    ' 2>/dev/null || echo "UNREACHABLE")
    printf "%-8s %-45s %s\n" "$host" "$short_model" "$result"
  done
}

cmd_logs() {
  local host="$1"
  local ip=$(get_ip "$host")
  ssh -i "$SSH_KEY" $SSH_OPTS "ubuntu@$ip" "tail -20 /home/ubuntu/fada-results/eval_run.log 2>/dev/null"
}

cmd_pull() {
  local host="$1"
  local ip=$(get_ip "$host")
  if [ -z "$ip" ]; then
    echo "Unknown host: $host" >&2; exit 1
  fi
  local dest="$SCRIPT_DIR/results/"
  mkdir -p "$dest"
  echo "Pulling checkpoint from $host ($ip)..."
  scp -i "$SSH_KEY" $SSH_OPTS "ubuntu@$ip:/home/ubuntu/fada-results/checkpoint_vllm_*.json" "$dest"
  echo "Done. Files in $dest"
  ls -lt "${dest}"checkpoint_vllm_*.json 2>/dev/null | head -3
}

cmd_vllm_log() {
  local host="$1"
  local ip=$(get_ip "$host")
  ssh -i "$SSH_KEY" $SSH_OPTS "ubuntu@$ip" "tail -30 /home/ubuntu/fada-results/vllm.log 2>/dev/null"
}

cmd_queue_log() {
  local host="$1"
  local ip=$(get_ip "$host")
  ssh -i "$SSH_KEY" $SSH_OPTS "ubuntu@$ip" "tail -30 /home/ubuntu/fada-results/queue.log 2>/dev/null"
}

case "${1:-help}" in
  play)      shift; cmd_play "$@" ;;
  ssh)       shift; cmd_ssh "$@" ;;
  status)    cmd_status ;;
  logs)      shift; cmd_logs "$@" ;;
  pull)      shift; cmd_pull "$@" ;;
  vllm-log)  shift; cmd_vllm_log "$@" ;;
  queue-log) shift; cmd_queue_log "$@" ;;
  help|*)
    echo "Usage: rccg.sh <command> [args]"
    echo "  play <playbook> [--limit hosts]  Run ansible-playbook"
    echo "  ssh <host> [cmd]                 SSH into host or run command"
    echo "  status                           Check eval progress"
    echo "  logs <host>                      Tail eval log"
    echo "  pull <host>                      Pull checkpoint files"
    echo "  vllm-log <host>                  Tail vLLM server log"
    echo "  queue-log <host>                 Tail model queue log"
    ;;
esac
