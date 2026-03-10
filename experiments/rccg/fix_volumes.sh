#!/bin/bash
HF="$HOME/.local/bin/hf"
TOKEN="$1"

# Check if the 8th disk is mounted
DISK8="/dev/disk/by-id/virtio-f997aec3-a855-4035-9"
if [ ! -b "$DISK8" ]; then
  echo "FATAL: 8th disk not found at $DISK8"
  exit 1
fi

# Mount it
MOUNT="/mnt/vol6"
if ! mountpoint -q "$MOUNT" 2>/dev/null; then
  echo "Formatting and mounting $DISK8 at $MOUNT"
  sudo mkfs.ext4 -F "$DISK8" > /dev/null 2>&1
  sudo mkdir -p "$MOUNT"
  sudo mount "$DISK8" "$MOUNT"
  sudo chown ubuntu:ubuntu "$MOUNT"
else
  echo "$MOUNT already mounted"
fi

# Download medgemma to its own volume
echo "Downloading medgemma-4b-it to $MOUNT..."
HF_HOME=${MOUNT}/huggingface HF_TOKEN="$TOKEN" $HF download "google/medgemma-4b-it" 2>&1 | tail -3
echo ""

# Remove medgemma from shared volume
echo "Removing medgemma from /mnt/volume..."
rm -rf /mnt/volume/huggingface/hub/models--google--medgemma-4b-it
echo "Done"

echo ""
echo "=== FINAL: ONE MODEL PER VOLUME ==="
for d in /mnt/volume /mnt/vol0 /mnt/vol1 /mnt/vol2 /mnt/vol3 /mnt/vol4 /mnt/vol5 /mnt/vol6; do
  SIZE=$(du -sh "$d/huggingface/hub/" 2>/dev/null | cut -f1)
  MODELS=$(ls "$d/huggingface/hub/" 2>/dev/null | grep "^models--" | sed 's/models--//g' | tr '\n' ' ')
  echo "$d: ${SIZE:-0}  $MODELS"
done
