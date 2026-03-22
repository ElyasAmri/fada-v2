#!/bin/bash
# Quick status check for all RCCG machines
# Runs parallel SSH to all hosts, completes within 30 seconds
# Usage: bash experiments/rccg/check_status.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INVENTORY="$SCRIPT_DIR/inventory/hosts.yml"
SSH_KEY="$HOME/.ssh/rccg_key"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o LogLevel=ERROR"

get_ip() {
  local host="$1"
  grep -A5 "^\s*${host}:" "$INVENTORY" | grep ansible_host | awk '{print $NF}' | head -1
}

get_hosts() {
  grep -oE 'fada-[0-9]+' "$INVENTORY" | sort -u -t- -k2 -n
}

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

# Launch parallel SSH to all hosts
for host in $(get_hosts); do
  ip=$(get_ip "$host")
  if [ -z "$ip" ]; then
    echo "UNREACHABLE" > "$TMPDIR/$host"
    continue
  fi

  (
    result=$(ssh -i "$SSH_KEY" $SSH_OPTS "ubuntu@$ip" 'bash -s' <<'REMOTE_CMD'
GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "-1")
GPU_MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
GPU_MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")

TRAIN_PID=$(pgrep -f '[t]rain_unsloth|[t]rain_qwen3vl|[r]un_queue|[l]aunch_fc' | head -1)
EVAL_PID=$(pgrep -f '[t]est_api_vlm|[e]val_hf_peft' | head -1)
SETUP_PID=$(pgrep -f '[a]pt-get|[d]pkg|[p]ip install|[p]ip3 install|[g]it clone|[w]get|[c]url.*huggingface' | head -1)

LOG=""
for f in /home/ubuntu/training_full.log /home/ubuntu/fada-results/eval_run.log; do
  [ -f "$f" ] && LOG="$f" && break
done

LOG_ERR=""
STEP=""
LOSS=""
JOB_EXIT=""
IS_EVAL=""
TRAIN_STEP=""
INFO=""
if [ -n "$LOG" ]; then
  TAIL=$(tail -c 5000 "$LOG" 2>/dev/null | tr '\r' '\n')
  LOG_ERR=$(echo "$TAIL" | grep -oE 'CUDA out of memory|OOM|RuntimeError|Traceback' | head -1)
  STEP=$(echo "$TAIL" | grep -oP '\d+/\d+' | tail -1)
  # Detect eval vs training: find largest-denominator step (training) and compare
  TRAIN_STEP=$(echo "$TAIL" | grep -oP '\d+/\d+' | awk -F/ '{print $0, $2}' | sort -k2 -n | tail -1 | awk '{print $1}')
  if [ -n "$STEP" ] && [ -n "$TRAIN_STEP" ]; then
    STEP_D=$(echo "$STEP" | cut -d/ -f2)
    TRAIN_D=$(echo "$TRAIN_STEP" | cut -d/ -f2)
    [ "$STEP_D" != "$TRAIN_D" ] && IS_EVAL="1"
  fi
  LOSS=$(echo "$TAIL" | grep -oP 'loss[=: ]+[0-9]+\.[0-9]+' | tail -1 | grep -oP '[0-9]+\.[0-9]+')
  # Check launch_job completion marker
  JOB_EXIT=$(echo "$TAIL" | grep -oP 'exit=\K[0-9]+' | tail -1)
fi

DONE=""
[ -f /home/ubuntu/fada-results/eval_done.marker ] && DONE="1"

TMUX_ALIVE=""
tmux has-session -t training 2>/dev/null && TMUX_ALIVE="1"

if [ -n "$TRAIN_PID" ]; then
  STATUS="TRAINING"
elif [ -n "$EVAL_PID" ]; then
  STATUS="EVAL"
elif [ -n "$SETUP_PID" ]; then
  STATUS="SETUP"
elif [ -n "$TMUX_ALIVE" ]; then
  STATUS="RUNNING"
elif [ -n "$JOB_EXIT" ]; then
  if [ "$JOB_EXIT" = "0" ]; then
    STATUS="DONE"
  else
    STATUS="FAILED(rc=$JOB_EXIT)"
  fi
elif [ -n "$DONE" ]; then
  STATUS="DONE"
elif [ -n "$LOG_ERR" ]; then
  STATUS="FAILED"
else
  STATUS="IDLE"
fi

# Build info string: show step/loss for training, or last log line for other jobs
if [ -n "$STEP" ] || [ -n "$LOSS" ]; then
  INFO=""
  if [ -n "$IS_EVAL" ]; then
    INFO="Eval:$STEP (Train:$TRAIN_STEP)"
  elif [ -n "$STEP" ]; then
    INFO="Step:$STEP"
  fi
  [ -n "$LOSS" ] && INFO="$INFO Loss:$LOSS"
elif [ -n "$LOG" ] && [ -z "$TRAIN_PID" ] && [ -z "$EVAL_PID" ]; then
  # For non-training jobs, show last meaningful log line
  INFO=$(echo "$TAIL" | grep -v '^\s*$' | grep -v '^\[launch_job\]' | tail -1 | head -c 40)
fi

echo "${STATUS}|${GPU_UTIL}|${GPU_MEM_USED}/${GPU_MEM_TOTAL}|${INFO}|${LOG_ERR}"
REMOTE_CMD
    2>/dev/null)

    if [ $? -ne 0 ] || [ -z "$result" ]; then
      echo "UNREACHABLE" > "$TMPDIR/$host"
    else
      echo "$result" > "$TMPDIR/$host"
    fi
  ) 2>/dev/null &
done

# Wait for all SSH jobs with 20s timeout
WAIT_END=$((SECONDS + 20))
while [ $SECONDS -lt $WAIT_END ] && jobs -r 2>/dev/null | grep -q .; do
  sleep 1
done

# Kill stragglers
for pid in $(jobs -p 2>/dev/null); do
  kill "$pid" 2>/dev/null
done
wait 2>/dev/null

# Load model names from jobs.json
JOBS_JSON="$SCRIPT_DIR/jobs.json"
get_model_short() {
  local host="$1"
  if [ -f "$JOBS_JSON" ]; then
    local cmd
    cmd=$(grep -A5 "\"$host\"" "$JOBS_JSON" | grep '"command"' | head -1)
    local model
    model=$(echo "$cmd" | grep -oE '\-\-model [^ ]+' | head -1 | sed 's/--model //')
    if [ -z "$model" ]; then
      model=$(echo "$cmd" | grep -oE 'eval_adapter\.sh [^ ]+' | head -1 | sed 's/eval_adapter.sh //')
    fi
    # Shorten model name
    echo "$model" | sed 's|.*/||; s|-Instruct||; s|-instruct||; s|unsloth-bnb-4bit||; s|unsloth/||'
  fi
}

# Output results
printf "%-10s %-15s %-8s %-20s %-25s %s\n" "HOST" "STATUS" "GPU" "MODEL" "INFO" "ERROR"
printf "%-10s %-15s %-8s %-20s %-25s %s\n" "----------" "---------------" "--------" "--------------------" "-------------------------" "-----"
for host in $(get_hosts); do
  if [ ! -f "$TMPDIR/$host" ]; then
    printf "%-10s %-15s\n" "$host" "UNREACHABLE"
    continue
  fi

  raw=$(cat "$TMPDIR/$host")
  if [ "$raw" = "UNREACHABLE" ]; then
    printf "%-10s %-15s\n" "$host" "UNREACHABLE"
    continue
  fi

  IFS='|' read -r status gpu_util gpu_mem info err <<< "$raw"
  gpu_str="GPU:${gpu_util}%"
  err_str=""
  [ -n "$err" ] && err_str="ERR:$err"
  model_str=$(get_model_short "$host")
  printf "%-10s %-15s %-8s %-20s %-25s %s\n" "$host" "$status" "$gpu_str" "$model_str" "$info" "$err_str"
done
