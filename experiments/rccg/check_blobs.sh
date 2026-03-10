#!/bin/bash
for d in /mnt/volume /mnt/vol0 /mnt/vol1 /mnt/vol2 /mnt/vol3 /mnt/vol4 /mnt/vol5; do
  SIZE=$(du -sh "$d/huggingface/hub/" 2>/dev/null | cut -f1)
  MODELS=$(ls "$d/huggingface/hub/" 2>/dev/null | grep "^models--" | sed 's/models--//g')
  echo "$d: $SIZE  [$MODELS]"
done
