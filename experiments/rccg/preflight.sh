#!/bin/bash
# Pre-flight validation before launching a training job.
# Checks disk, framework, data, GPU, and conflicting processes.
# Usage: preflight.sh [framework] [data_root]
#   framework: llamafactory|unsloth|swift (default: auto-detect from args)
#   data_root: path to image directory (default: /home/ubuntu/fada-v3/data/Fetal Ultrasound)
#
# Outputs JSON to stdout: {"ok": true} or {"ok": false, "errors": [...]}
# Exit 0 = pass, Exit 1 = fail

set -euo pipefail

FRAMEWORK="${1:-}"
DATA_ROOT="${2:-/home/ubuntu/fada-v3/data/Fetal Ultrasound}"
MIN_FREE_GB=20
ERRORS=()

# --- Disk space ---
free_kb=$(df / --output=avail | tail -1 | tr -d ' ')
free_gb=$((free_kb / 1024 / 1024))

# Check if volume is mounted (most storage goes there)
vol_free_gb=0
if mountpoint -q /mnt/data 2>/dev/null; then
    vol_free_kb=$(df /mnt/data --output=avail | tail -1 | tr -d ' ')
    vol_free_gb=$((vol_free_kb / 1024 / 1024))
fi

if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
    echo "Root disk low (${free_gb}GB free). Cleaning caches..." >&2
    rm -rf /home/ubuntu/.cache/huggingface/hub/models--* 2>/dev/null || true
    rm -rf /home/ubuntu/.cache/pip 2>/dev/null || true
    rm -rf /tmp/torch* /tmp/tiktoken* 2>/dev/null || true

    free_kb=$(df / --output=avail | tail -1 | tr -d ' ')
    free_gb=$((free_kb / 1024 / 1024))
    # Only fail if root is critically low AND no volume available
    if [ "$free_gb" -lt 5 ] && [ "$vol_free_gb" -lt 20 ]; then
        ERRORS+=("Disk critically low: root=${free_gb}GB, volume=${vol_free_gb}GB")
    else
        echo "Disk OK: root=${free_gb}GB, volume=${vol_free_gb}GB" >&2
    fi
fi

# --- Framework availability ---
if [ -n "$FRAMEWORK" ]; then
    case "$FRAMEWORK" in
        llamafactory)
            if ! python3 -c "import llamafactory" 2>/dev/null; then
                ERRORS+=("llamafactory not importable: pip3 install llamafactory")
            fi
            ;;
        unsloth)
            if ! python3 -c "import unsloth" 2>/dev/null; then
                ERRORS+=("unsloth not importable: pip3 install unsloth")
            fi
            ;;
        swift)
            if ! python3 -c "import swift" 2>/dev/null; then
                ERRORS+=("ms-swift not importable: pip3 install ms-swift")
            fi
            ;;
    esac
fi

# --- Data files ---
TRAIN_JSONL="/home/ubuntu/fada-v3/data/vlm_training/gt_train.jsonl"
if [ ! -s "$TRAIN_JSONL" ]; then
    # Check ShareGPT variant
    TRAIN_JSONL="/home/ubuntu/fada-v3/data/vlm_training/gt_train_sharegpt_abs.jsonl"
    if [ ! -s "$TRAIN_JSONL" ]; then
        ERRORS+=("Training JSONL not found or empty")
    fi
fi

if [ ! -d "$DATA_ROOT" ]; then
    ERRORS+=("Data root directory not found: $DATA_ROOT")
fi

# --- GPU health ---
if ! nvidia-smi > /dev/null 2>&1; then
    ERRORS+=("nvidia-smi failed: GPU may be unhealthy")
fi

# --- Conflicting processes ---
if pgrep -f 'train_unsloth|train_llamafactory|train_swift|swift sft|llamafactory-cli train' > /dev/null 2>&1; then
    ERRORS+=("Conflicting training process already running")
fi

# --- Output ---
if [ ${#ERRORS[@]} -eq 0 ]; then
    echo '{"ok": true}'
    exit 0
else
    # Build JSON array of errors
    json_errors=""
    for err in "${ERRORS[@]}"; do
        [ -n "$json_errors" ] && json_errors+=","
        json_errors+="\"$(echo "$err" | sed 's/"/\\"/g')\""
    done
    echo "{\"ok\": false, \"errors\": [$json_errors]}"
    exit 1
fi
