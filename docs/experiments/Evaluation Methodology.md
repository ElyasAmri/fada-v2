---
tags: [phase1, methods]
---
**Phase**: 1 - VLM Benchmarking

## Hardware Setup
- **GPU**: NVIDIA RTX 4070 Laptop (8GB VRAM)
- **OS**: Windows 11
- **Framework**: PyTorch 2.8.0+cu128, Transformers 4.x

## Quantization Strategy
```
Method: 4-bit NF4 with BitsAndBytes
Applied to: Models >6GB
Config:
  - bnb_4bit_quant_type: "nf4"
  - bnb_4bit_compute_dtype: torch.bfloat16
  - bnb_4bit_use_double_quant: True
```

## Test Dataset
- **Source**: data/Fetal Ultrasound/
- **Categories**: 7 organ types
- **Images per test**: 6-8 images
- **Format**: PNG ultrasound scans

## Evaluation Metrics

### 1. Fetal Context Recognition
Checks for keywords: fetal, fetus, ultrasound, pregnancy, prenatal

### 2. Anatomy Identification
Category-specific term matching (abdomen -> stomach, liver, kidney, etc.)

### 3. Medical Terminology Usage
Advanced terms: structure, anatomical, echogenic, etc.

### 4. Overall Score
Combined weighted average against BLIP-2 baseline (~55%)

## Quantization Success Rates
- **4-bit NF4**: 95% success on compatible models
- **8-bit**: 90% success
- **Full precision**: Limited to <6GB models

## Links
- [[VLM Models Tested]] - Models evaluated
- [[VLM Testing Results]] - Results

