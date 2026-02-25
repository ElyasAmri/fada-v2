---
tags: [phase1, methods]
---

**Covers**: Phase 1 (proxy) and Phase 2/3 (ground truth) evaluation

## Hardware

### Phase 1 (October 2025)

- **GPU**: NVIDIA RTX 4070 Laptop (8GB VRAM)
- **OS**: Windows 11
- **Framework**: PyTorch 2.8.0+cu128, Transformers 4.x

### Phase 2+ (January 2026+)

- **Local GPU**: NVIDIA RTX 5090 (24GB VRAM)
- **Cloud**: vast.ai (RTX 3090/4090/A100), RunPod (RTX 3090/4090)

## Phase 1: Proxy Metrics (DEPRECATED for comparison)

Used keyword matching on ~250 samples against BLIP-2 baseline. These scores are **not comparable** to Phase 2/3 ground-truth scores.

### Quantization Strategy

```
Method: 4-bit NF4 with BitsAndBytes
Applied to: Models >6GB
Config:
  - bnb_4bit_quant_type: "nf4"
  - bnb_4bit_compute_dtype: torch.bfloat16
  - bnb_4bit_use_double_quant: True
```

### Proxy Evaluation Metrics

1. **Fetal Context Recognition** - keyword matching (fetal, fetus, ultrasound, etc.)
2. **Anatomy Identification** - category-specific term matching
3. **Medical Terminology Usage** - advanced term detection
4. **Overall Score** - combined weighted average against BLIP-2 baseline

## Phase 2/3: Ground Truth Evaluation (CURRENT)

Uses embedding similarity scoring with sentence-transformers against expert annotations.

### Method

- Compare model VQA response text to ground truth annotation text
- Sentence-transformers embedding similarity
- Per-question scoring (Q1-Q8 independently)
- Per-category scoring (14 anatomical categories)

### Dataset

- **Full dataset**: ~19,000 images, 14 classes
- **Test set**: 1,494 images
- **Train set**: 12,014 images
- **Val set**: 1,494 images
- **Questions per image**: 8 (Q1-Q8)

### Verified Results

| Model                      | Score  | Samples | Method               |
| -------------------------- | ------ | ------- | -------------------- |
| Qwen2.5-VL-7B (fine-tuned) | 81.1%  | 600     | Embedding similarity |
| MedGemma-27B (cloud API)   | 78.81% | 709     | Embedding similarity |

### Known Issues

- Brain sub-views (Trans-cerebellum, Trans-thalamic, Trans-ventricular) score ~57-58% consistently
- Q7 has 95% "normal" class imbalance
- Embedding similarity not yet validated against human expert judgment
