#!/bin/bash
# Stop hook: blocks Claude Code until a remote SSH process finishes.
# Only activates when wait_marker.json exists. Otherwise instant exit (zero overhead).
#
# Marker format:
#   {"host":"fada-1","process_pattern":"eval_hf_peft","log_file":"/home/ubuntu/eval.log","poll_interval":60,"max_checks":500}
#
# On completion, writes wait_result.json with final status and log tail.
# Exits with code 2 + stdout message to trigger Claude Code continuation.

set -uo pipefail

HOOK_DIR="$(cd "$(dirname "$0")" && pwd)"
MARKER="$HOOK_DIR/wait_marker.json"
RESULT="$HOOK_DIR/wait_result.json"

# Fast exit if no marker (normal operation, zero overhead)
if [ ! -f "$MARKER" ]; then
  exit 0
fi

# Parse marker fields (no jq -- Git Bash on Windows may lack it)
parse_field() {
  grep -o "\"$1\"[[:space:]]*:[[:space:]]*\"[^\"]*\"" "$MARKER" 2>/dev/null | sed "s/.*\"$1\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\".*/\1/"
}
parse_int() {
  grep -o "\"$1\"[[:space:]]*:[[:space:]]*[0-9]*" "$MARKER" 2>/dev/null | sed "s/.*:[[:space:]]*//"
}

HOST=$(parse_field host)
PROCESS=$(parse_field process_pattern)
LOG_FILE=$(parse_field log_file)
POLL_INTERVAL=$(parse_int poll_interval)
MAX_CHECKS=$(parse_int max_checks)

# Defaults
LOG_FILE="${LOG_FILE:-/home/ubuntu/eval.log}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"
MAX_CHECKS="${MAX_CHECKS:-500}"

if [ -z "$HOST" ] || [ -z "$PROCESS" ]; then
  echo '{"status":"error","message":"Marker missing host or process_pattern"}' > "$RESULT"
  rm -f "$MARKER"
  echo "Remote process monitor: marker missing host or process_pattern"
  exit 2
fi

# Validate PROCESS: strict allowlist (alphanumeric, dots, hyphens, underscores, min 2 chars)
if ! echo "$PROCESS" | grep -qE '^[a-zA-Z0-9._-]{2,}$'; then
  echo '{"status":"error","message":"Invalid process_pattern: must be 2+ alphanumeric/dot/hyphen/underscore chars"}' > "$RESULT"
  rm -f "$MARKER"
  echo "Remote process monitor: invalid process_pattern"
  exit 2
fi

# Validate LOG_FILE: strict allowlist (absolute path, alphanumeric/slashes/dots/hyphens/underscores/spaces)
if ! echo "$LOG_FILE" | grep -qE '^/[a-zA-Z0-9/_. -]+$'; then
  echo '{"status":"error","message":"Invalid log_file path: contains disallowed characters"}' > "$RESULT"
  rm -f "$MARKER"
  echo "Remote process monitor: invalid log_file path"
  exit 2
fi

# SSH config (reuse rccg.sh patterns)
PROJECT_DIR="$(cd "$HOOK_DIR/../.." && pwd)"
INVENTORY="$PROJECT_DIR/experiments/rccg/inventory/hosts.yml"
SSH_KEY="$HOME/.ssh/rccg_key"
SSH_OPTS_ARRAY=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o LogLevel=ERROR)

# Resolve host IP from inventory
IP=$(grep -A1 "^        $HOST:" "$INVENTORY" 2>/dev/null | grep ansible_host | awk '{print $2}')
if [ -z "$IP" ]; then
  echo "{\"status\":\"error\",\"message\":\"Unknown host: $HOST\"}" > "$RESULT"
  rm -f "$MARKER"
  echo "Remote process monitor: unknown host $HOST"
  exit 2
fi

echo "[wait-for-remote] Watching '$PROCESS' on $HOST ($IP), polling every ${POLL_INTERVAL}s (max $MAX_CHECKS checks)" >&2

# Poll loop
CHECKS=0
UNREACHABLE_COUNT=0
MAX_UNREACHABLE=3

while true; do
  CHECKS=$((CHECKS + 1))

  # Circuit breaker: max poll duration
  if [ "$CHECKS" -gt "$MAX_CHECKS" ]; then
    py -c "
import json
result = {'status': 'timed_out', 'host': '$HOST', 'process': '$PROCESS', 'checks': $CHECKS, 'max_checks': $MAX_CHECKS}
print(json.dumps(result))
" > "$RESULT"
    rm -f "$MARKER"
    echo "[wait-for-remote] Timed out after $MAX_CHECKS checks." >&2
    echo "Remote process '$PROCESS' on $HOST timed out after $MAX_CHECKS checks. Results in .claude/hooks/wait_result.json"
    exit 2
  fi

  # Bracket trick: wrap first char in [] so pgrep doesn't match itself or the SSH shell
  BRACKET_PATTERN="[${PROCESS:0:1}]${PROCESS:1}"
  RUNNING=$(ssh -i "$SSH_KEY" "${SSH_OPTS_ARRAY[@]}" "ubuntu@$IP" \
    "pgrep -f '$BRACKET_PATTERN' > /dev/null 2>&1 && echo yes || echo no" \
    2>/dev/null)
  SSH_RC=$?

  if [ $SSH_RC -ne 0 ]; then
    UNREACHABLE_COUNT=$((UNREACHABLE_COUNT + 1))
    echo "[wait-for-remote] SSH failed (attempt $UNREACHABLE_COUNT/$MAX_UNREACHABLE)" >&2
    if [ $UNREACHABLE_COUNT -ge $MAX_UNREACHABLE ]; then
      echo "{\"status\":\"unreachable\",\"host\":\"$HOST\",\"checks\":$CHECKS}" > "$RESULT"
      rm -f "$MARKER"
      echo "[wait-for-remote] Host $HOST unreachable after $MAX_UNREACHABLE consecutive failures." >&2
      echo "Remote process monitor: $HOST unreachable after $MAX_UNREACHABLE failures"
      exit 2
    fi
    sleep "$POLL_INTERVAL"
    continue
  fi

  # Reset unreachable counter on successful SSH
  UNREACHABLE_COUNT=0

  if [ "$RUNNING" = "no" ]; then
    # Process finished -- capture final log
    FINAL_LOG=$(ssh -i "$SSH_KEY" "${SSH_OPTS_ARRAY[@]}" "ubuntu@$IP" \
      "tail -30 '$LOG_FILE'" 2>/dev/null || echo "(could not read log)")

    # Use python for safe JSON construction (handles \t, \r, control chars, quotes, backslashes)
    py -c "
import json, sys
log = sys.stdin.read()
result = {
    'status': 'completed',
    'host': '$HOST',
    'process': '$PROCESS',
    'checks': $CHECKS,
    'log': log
}
print(json.dumps(result))
" <<< "$FINAL_LOG" > "$RESULT"

    rm -f "$MARKER"
    echo "[wait-for-remote] Process '$PROCESS' on $HOST completed after $CHECKS checks." >&2
    echo "Remote process '$PROCESS' on $HOST completed after $CHECKS checks. Results in .claude/hooks/wait_result.json"
    exit 2
  fi

  # Still running
  if [ $((CHECKS % 5)) -eq 0 ]; then
    echo "[wait-for-remote] Still running on $HOST (check #$CHECKS)" >&2
  fi

  sleep "$POLL_INTERVAL"
done
