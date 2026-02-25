---
tags: [phase2, training]
---
**Phase**: 2 - Active Development

## Strategy
Fine-tune top-performing VLMs on fetal ultrasound dataset using LoRA adapters for efficient training on consumer GPU.

## Target Models (Priority Order)
1. **MiniCPM-V-2.6** - 88.9% baseline, primary target
2. **Qwen2-VL-2B** - 83.3% baseline, efficient alternative
3. **InternVL2-4B** - 82% baseline, strong medical understanding

## Training Configuration

### LoRA Parameters
```python
lora_config = {
    "r": 8,
    "alpha": 16,
    "target_modules": ["q_proj", "v_proj"],
    "dropout": 0.05
}
```

### Training Setup
- **Quantization**: 8-bit (LoRA training)
- **Learning Rate**: 1e-4
- **Batch Size**: 1-4 (GPU constrained)
- **Epochs**: 3-5 for full training

## Expected Outcomes
- **Current baseline**: 88.9% (zero-shot)
- **Target after fine-tuning**: 95%+
- **Training time**: ~1 hour per model

## Infrastructure
Training executed on vast.ai cloud GPUs for larger batch sizes and faster iteration.

See [[Vastai CLI Implementation]] for cloud infrastructure.

## Progress

### Phase 2a: Local Validation
- [x] LoRA pipeline validated on BLIP-2
- [x] 5 organ categories trained (1 epoch each)
- [x] Generation parameters optimized

### Phase 2b: Cloud Training (In Progress)
- [ ] Deploy to vast.ai
- [ ] Full dataset training
- [ ] Hyperparameter sweep

## Links
- [[VLM Testing Results]] - Baseline performance
- [[Vastai CLI Implementation]] - Cloud infrastructure
- [[Literature Review]] - Research backing

