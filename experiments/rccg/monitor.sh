#!/bin/bash
# Monitor eval progress across all RCCG machines
# Polls every 2 minutes, reports when machines finish

KEY=~/.ssh/rccg_key
SSH_OPTS="-i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o LogLevel=ERROR"
RESULTS_DIR=/home/ubuntu/fada-results
INTERVAL=120

declare -A HOSTS
HOSTS=(
  [fada-1]="62.169.159.176|Qwen/Qwen2.5-VL-3B-Instruct"
  [fada-2]="185.216.20.70|OpenGVLab/InternVL3_5-8B"
  [fada-3]="185.216.21.62|llava-hf/llava-onevision-qwen2-7b-ov-hf"
  [fada-4]="38.128.232.129|microsoft/Phi-4-multimodal-instruct"
  [fada-5]="38.128.232.145|Qwen/Qwen3-VL-4B-Instruct"
  [fada-6]="38.128.232.131|openbmb/MiniCPM-V-4_5"
  [fada-7]="69.19.137.14|google/gemma-3-12b-it"
  [fada-8]="69.19.137.100|google/medgemma-4b-it"
)

declare -A DONE

echo "=== RCCG Eval Monitor ==="
echo "Polling every ${INTERVAL}s. Press Ctrl+C to stop."
echo ""

while true; do
  TIMESTAMP=$(date '+%H:%M:%S')
  ALL_DONE=true

  for HOST in $(echo "${!HOSTS[@]}" | tr ' ' '\n' | sort); do
    IFS='|' read -r IP MODEL <<< "${HOSTS[$HOST]}"

    # Skip already-done machines
    if [[ "${DONE[$HOST]}" == "1" ]]; then
      continue
    fi

    # Check done marker first, then progress
    STATUS=$(ssh $SSH_OPTS ubuntu@$IP "
      if [ -f $RESULTS_DIR/eval_done.marker ]; then
        echo 'DONE'
        cat $RESULTS_DIR/eval_done.marker
      elif pgrep -f '[t]est_api_vlm.py' > /dev/null 2>&1; then
        LAST=\$(tail -1 $RESULTS_DIR/eval_run.log 2>/dev/null | grep -oP '\d+%.*?it/s' | tail -1)
        echo \"RUNNING \$LAST\"
      else
        echo 'NOT_RUNNING'
      fi
    " 2>/dev/null)

    if echo "$STATUS" | grep -q "^DONE"; then
      EXIT_CODE=$(echo "$STATUS" | grep exit_code | cut -d= -f2)
      DONE[$HOST]=1
      if [ "$EXIT_CODE" = "0" ]; then
        echo "[$TIMESTAMP] *** $HOST FINISHED ($MODEL) ***"
      else
        echo "[$TIMESTAMP] *** $HOST FAILED (exit=$EXIT_CODE, $MODEL) ***"
      fi
    else
      ALL_DONE=false
      echo "[$TIMESTAMP] $HOST: $STATUS"
    fi
  done

  if $ALL_DONE; then
    echo ""
    echo "=== ALL MACHINES DONE ==="
    break
  fi

  echo ""
  sleep $INTERVAL
done
