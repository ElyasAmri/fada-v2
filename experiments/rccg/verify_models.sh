#!/bin/bash
echo "=== MODEL VERIFICATION ==="
declare -A VOL_MODELS
VOL_MODELS[/mnt/volume]="Qwen--Qwen2.5-VL-3B-Instruct google--medgemma-4b-it"
VOL_MODELS[/mnt/vol0]="OpenGVLab--InternVL3_5-8B"
VOL_MODELS[/mnt/vol1]="llava-hf--llava-onevision-qwen2-7b-ov-hf"
VOL_MODELS[/mnt/vol2]="microsoft--Phi-4-multimodal-instruct"
VOL_MODELS[/mnt/vol3]="deepseek-ai--deepseek-vl2-small"
VOL_MODELS[/mnt/vol4]="openbmb--MiniCPM-V-4_5"
VOL_MODELS[/mnt/vol5]="google--gemma-3-12b-it"

ALL_OK=true
for MOUNT in /mnt/volume /mnt/vol0 /mnt/vol1 /mnt/vol2 /mnt/vol3 /mnt/vol4 /mnt/vol5; do
  for MODEL in ${VOL_MODELS[$MOUNT]}; do
    SNAP_DIR="$MOUNT/huggingface/hub/models--${MODEL}/snapshots"
    if [ -d "$SNAP_DIR" ]; then
      SNAP=$(ls "$SNAP_DIR" | head -1)
      FILE_COUNT=$(find "$SNAP_DIR/$SNAP" -type f | wc -l)
      SIZE=$(du -sh "$SNAP_DIR/$SNAP" | cut -f1)
      # Check for config.json (every model must have it)
      if [ -f "$SNAP_DIR/$SNAP/config.json" ]; then
        echo "OK   $MOUNT  $MODEL  ${FILE_COUNT} files  ${SIZE}"
      else
        echo "FAIL $MOUNT  $MODEL  no config.json"
        ALL_OK=false
      fi
    else
      echo "FAIL $MOUNT  $MODEL  no snapshots dir"
      ALL_OK=false
    fi
  done
done

echo ""
if $ALL_OK; then
  echo "ALL MODELS VERIFIED"
else
  echo "SOME MODELS FAILED"
fi
