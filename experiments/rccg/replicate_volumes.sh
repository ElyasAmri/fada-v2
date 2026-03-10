#!/bin/bash
MASTER=/mnt/volume/huggingface

echo "Master volume: $(du -sh $MASTER | cut -f1)"
echo "Replicating to 7 volumes sequentially..."
echo ""

for i in 0 1 2 3 4 5 6; do
  DEST="/mnt/vol${i}"
  if ! mountpoint -q "$DEST" 2>/dev/null; then
    echo "SKIP vol${i}: not mounted"
    continue
  fi

  echo "=== vol${i} ==="
  # Wipe old single-model cache
  rm -rf "${DEST}/huggingface" "${DEST}/download.log"
  # Rsync master to this volume
  rsync -a --info=progress2 "$MASTER" "${DEST}/"
  echo "  Done: $(du -sh ${DEST}/huggingface | cut -f1)"
  echo ""
done

echo "=== ALL DONE ==="
df -h /mnt/volume /mnt/vol0 /mnt/vol1 /mnt/vol2 /mnt/vol3 /mnt/vol4 /mnt/vol5 /mnt/vol6
