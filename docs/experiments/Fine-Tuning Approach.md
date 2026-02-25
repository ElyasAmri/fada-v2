---
tags: [phase2, training]
---

**Phase**: 2 - Complete

## Strategy

Fine-tune top-performing VLMs on fetal ultrasound dataset using LoRA adapters for efficient training.

## Completed Results

| Model                 | Score | Method               | Samples | Platform        |
| --------------------- | ----- | -------------------- | ------- | --------------- |
| Qwen2.5-VL-7B         | 81.1% | Embedding similarity | 600     | RunPod RTX 3090 |
| Qwen3-VL-8B (Q7 only) | 82%   | Accuracy             | 50      | Local RTX 5090  |

## Target Models (Priority Order)

1. **Qwen2.5-VL-7B** - 81.1% fine-tuned, best verified GT score
2. **MiniCPM-V-2.6** - 88.9% proxy (needs GT re-evaluation)
3. **Qwen2-VL-2B** - 83.3% proxy, efficient 2B model
4. **InternVL2-4B** - ~82% proxy, strong medical understanding

Note: Proxy scores (keyword matching, ~250 samples) are NOT comparable to GT scores (embedding similarity).

## Training Configuration

### LoRA Parameters (Current)

```python
lora_config = {
    "r": 16,
    "alpha": 32,
    "target_modules": "all_linear",
    "dropout": 0.05
}
```

### Training Setup

- **Quantization**: 8-bit (LoRA training)
- **Learning Rate**: 1e-4
- **Batch Size**: 1-4 (GPU constrained)
- **Epochs**: 1 for benchmarking, 3 for final

## Infrastructure

- **Local**: RTX 5090 (24GB VRAM) with Unsloth (verified for all 7 Qwen models)
- **Cloud**: vast.ai and RunPod (RTX 3090/4090 at ~$0.40/h)

### Unsloth-Compatible Models

- Qwen2-VL-2B, Qwen2-VL-7B
- Qwen2.5-VL-3B, Qwen2.5-VL-7B
- Qwen3-VL-2B, Qwen3-VL-4B, Qwen3-VL-8B

## Remaining Experiments

See `docs/experiments/unsloth_vlm_experiments.md` for 7 planned experiments:

1. Multi-task Q1-Q8 training
2. Combined comprehensive report generation
3. Layer ablation study
4. Model size comparison (2B vs 4B vs 8B)
5. Prompt engineering variations
6. Data augmentation impact
7. Class-balanced training (95% normal vs 5% abnormal)
