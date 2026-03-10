#!/bin/bash
cd /home/ubuntu/fada-v3
PYTHON=venv/bin/python
$PYTHON -c "
from huggingface_hub import snapshot_download
token='REDACTED_HF_TOKEN'
cache='/mnt/models/huggingface/hub'
for m in ['lmms-lab/LLaVA-OneVision-1.5-8B-Instruct','lmms-lab/LLaVA-OneVision-1.5-4B-Instruct','JZPeterPan/MedVLM-R1']:
    print(f'=== {m} ===')
    snapshot_download(m, cache_dir=cache, token=token)
    print('OK')
print('DONE')
"
