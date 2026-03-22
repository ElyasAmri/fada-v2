#!/bin/bash
# Launch a job on a remote RCCG machine via tmux
#
# Usage:
#   bash launch_job.sh <host> "<command>"       Write command to remote script, run in tmux
#   bash launch_job.sh <host> -f <script.sh>    Copy local script to remote, run in tmux
#
# The command/script runs inside a tmux session named 'training' that
# survives SSH disconnects. Output is logged to /home/ubuntu/training_full.log.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INVENTORY="$SCRIPT_DIR/inventory/hosts.yml"
SSH_KEY="$HOME/.ssh/rccg_key"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o LogLevel=ERROR"
JOBS_FILE="$SCRIPT_DIR/jobs.json"
REMOTE_SCRIPT="/home/ubuntu/job_cmd.sh"
REMOTE_LOG="/home/ubuntu/training_full.log"

get_ip() {
  local host="$1"
  grep -A5 "^\s*${host}:" "$INVENTORY" | grep ansible_host | awk '{print $NF}' | head -1
}

ssh_cmd() {
  ssh -i "$SSH_KEY" $SSH_OPTS "ubuntu@$IP" "$@"
}

if [ $# -lt 2 ]; then
  echo "Usage: launch_job.sh <host> \"<command>\""
  echo "       launch_job.sh <host> -f <script.sh>"
  exit 1
fi

HOST="$1"; shift
IP=$(get_ip "$HOST")

if [ -z "$IP" ]; then
  echo "Error: could not resolve IP for host '$HOST'" >&2
  exit 1
fi

# Build local temp script
TMPFILE=$(mktemp)
trap "rm -f $TMPFILE" EXIT

if [ "${1:-}" = "-f" ]; then
  # Copy user's script file with traceability wrapper
  SCRIPT_FILE="${2:?Missing script path}"
  if [ ! -f "$SCRIPT_FILE" ]; then
    echo "Error: script not found: $SCRIPT_FILE" >&2
    exit 1
  fi
  COMMAND="(script) $SCRIPT_FILE"
  {
    echo '#!/bin/bash'
    echo 'echo "[launch_job] Started at $(date -u)"'
    echo "echo \"[launch_job] Script: $SCRIPT_FILE\""
    echo 'echo "---"'
    cat "$SCRIPT_FILE"
    echo ''
    echo 'EXIT_CODE=$?'
    echo 'echo "---"'
    echo 'echo "[launch_job] Finished at $(date -u) exit=$EXIT_CODE"'
    echo 'exit $EXIT_CODE'
  } > "$TMPFILE"
else
  # Command string -- write it verbatim to a file (no shell expansion)
  COMMAND="$1"
  {
    echo '#!/bin/bash'
    echo 'echo "[launch_job] Started at $(date -u)"'
    echo 'echo "---"'
    printf '%s\n' "$COMMAND"
    echo 'EXIT_CODE=$?'
    echo 'echo "---"'
    echo 'echo "[launch_job] Finished at $(date -u) exit=$EXIT_CODE"'
    echo 'exit $EXIT_CODE'
  } > "$TMPFILE"
fi

echo "Launching job on $HOST ($IP)..."
echo "Command: $COMMAND"

# Detect framework from command
FRAMEWORK=""
case "$COMMAND" in
  *train_llamafactory*) FRAMEWORK="llamafactory" ;;
  *train_swift*|*swift\ sft*) FRAMEWORK="swift" ;;
  *train_unsloth*) FRAMEWORK="unsloth" ;;
esac

# Pre-flight validation
PREFLIGHT_SCRIPT="$SCRIPT_DIR/preflight.sh"
if [ -f "$PREFLIGHT_SCRIPT" ]; then
  scp -i "$SSH_KEY" $SSH_OPTS "$PREFLIGHT_SCRIPT" "ubuntu@$IP:/tmp/preflight.sh" >/dev/null
  PREFLIGHT_RESULT=$(ssh_cmd "bash /tmp/preflight.sh '$FRAMEWORK'" 2>/dev/null) || true
  if echo "$PREFLIGHT_RESULT" | grep -q '"ok": false'; then
    echo "Pre-flight FAILED: $PREFLIGHT_RESULT" >&2
    exit 1
  fi
  echo "Pre-flight passed"
fi

# SCP script to remote (byte-for-byte, no quoting issues)
scp -i "$SSH_KEY" $SSH_OPTS "$TMPFILE" "ubuntu@$IP:$REMOTE_SCRIPT" >/dev/null

# Kill existing tmux session, start new one
ssh_cmd "tmux kill-session -t training 2>/dev/null || true"
ssh_cmd "chmod +x $REMOTE_SCRIPT && tmux new-session -d -s training 'bash $REMOTE_SCRIPT 2>&1 | tee $REMOTE_LOG'"

# Wait for process to start
sleep 3

# Verify tmux session exists
if ssh_cmd "tmux has-session -t training 2>/dev/null"; then
  PANE_PID=$(ssh_cmd "tmux list-panes -t training -F '#{pane_pid}'" 2>/dev/null || echo "unknown")
  echo "Job running in tmux session 'training' (pane PID: $PANE_PID)"
else
  # Session gone -- check if job already completed (fast jobs)
  LAST_LINE=$(ssh_cmd "tail -1 $REMOTE_LOG 2>/dev/null" || echo "")
  if echo "$LAST_LINE" | grep -q "exit=0"; then
    PANE_PID="completed"
    echo "Job already completed successfully (fast job)"
  else
    echo "Error: tmux session not found and job did not complete cleanly" >&2
    ssh_cmd "tail -20 $REMOTE_LOG 2>/dev/null" || true
    exit 1
  fi
fi

# Update jobs.json
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
[ -f "$JOBS_FILE" ] || echo "{}" > "$JOBS_FILE"

PYTHON=""
for p in "$SCRIPT_DIR/../../venv/Scripts/python.exe" python3 python; do
  command -v "$p" &>/dev/null && PYTHON="$p" && break
done
if [ -z "$PYTHON" ]; then
  echo "Warning: python not found, skipping jobs.json update" >&2
else
JOBS_FILE_PY=$(cygpath -m "$JOBS_FILE" 2>/dev/null || echo "$JOBS_FILE")
JOB_HOST="$HOST" JOB_CMD="$COMMAND" JOB_TIME="$TIMESTAMP" JOB_PID="$PANE_PID" JOB_FILE="$JOBS_FILE_PY" \
  "$PYTHON" -c "
import json, os
jobs_file = os.environ['JOB_FILE']
with open(jobs_file) as f:
    jobs = json.load(f)
jobs[os.environ['JOB_HOST']] = {
    'command': os.environ['JOB_CMD'],
    'launched_at': os.environ['JOB_TIME'],
    'status': 'running',
    'pane_pid': os.environ['JOB_PID'],
}
with open(jobs_file, 'w') as f:
    json.dump(jobs, f, indent=2)
print(f'Updated {jobs_file}')
"
fi

echo "Done."
