#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/mnt/models/huggingface

eval "$(/home/ubuntu/miniconda3/bin/conda shell.bash hook)"
conda activate ft-unsloth

cd /home/ubuntu/fada-v3

# Run 50 steps and measure throughput
python -c "
import time, json, torch
from pathlib import Path

# Quick profiling: measure data loading vs training time
from experiments.framework_comparison.wrappers.train_unsloth import VLMDataset, collate_fn

with open('experiments/framework_comparison/config.json') as f:
    config = json.load(f)

from unsloth import FastModel

model, processor = FastModel.from_pretrained(
    'Qwen/Qwen2.5-VL-7B-Instruct',
    max_seq_length=4096,
    load_in_4bit=True,
    dtype=torch.bfloat16,
)

# Check attention implementation
print('Model attention:', getattr(model.config, 'attn_implementation', 'unknown'))

ds = VLMDataset('data/vlm_training/gt_train.jsonl', processor, max_samples=200, max_length=4096, data_root='data/Fetal Ultrasound')

from torch.utils.data import DataLoader
dl = DataLoader(ds, batch_size=16, collate_fn=collate_fn, num_workers=0)

# Time data loading
t0 = time.time()
batches = []
for i, batch in enumerate(dl):
    batches.append(batch)
    if i >= 2:
        break
data_time = time.time() - t0
print(f'Data loading (3 batches): {data_time:.1f}s ({data_time/3:.1f}s/batch)')

# Time a forward pass
batch = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in batches[0].items()}
t0 = time.time()
with torch.no_grad():
    out = model(**batch)
fwd_time = time.time() - t0
print(f'Forward pass: {fwd_time:.1f}s')
print(f'GPU memory: {torch.cuda.max_memory_allocated()/1e9:.1f}GB')
"
conda deactivate
