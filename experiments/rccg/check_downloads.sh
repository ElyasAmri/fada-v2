#!/bin/bash
echo "=== DOWNLOAD PROGRESS ==="
for f in /mnt/volume/download.log /mnt/vol0/download.log /mnt/vol1/download.log /mnt/vol2/download.log /mnt/vol3/download.log /mnt/vol4/download.log /mnt/vol5/download.log; do
  echo "--- $f ---"
  tail -1 "$f" 2>/dev/null || echo "  no log"
done
echo ""
echo "=== DISK USAGE ==="
df -h /mnt/volume /mnt/vol0 /mnt/vol1 /mnt/vol2 /mnt/vol3 /mnt/vol4 /mnt/vol5 2>/dev/null
echo ""
echo "=== PROCESSES ==="
ps aux | grep "[h]f download" || echo "none running"
echo ""
echo "=== MEMORY ==="
free -h | head -2
