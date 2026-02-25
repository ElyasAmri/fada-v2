---
tags: [tasks]
---

# Active Tasks

## Current Phase: 4 - Full Test Set Evaluation

### In Progress

- [ ] Ground-truth evaluation of top Phase 1 models on 1,494 test images
- [ ] Expand Qwen2.5-VL-7B evaluation from 600 to 1,494 images
- [ ] Score existing Gemini batch annotation results (62 batches)
- [ ] Score MedGemma-4B results (medgemma_4b_complete.json)

### Completed (Phase 1: VLM Benchmarking)

- [x] VLM benchmark framework
- [x] Test 50+ models with proxy metrics
- [x] Identify top performers (MiniCPM-V-2.6, Qwen2-VL-2B, InternVL2-4B)
- [x] Document results

### Completed (Phase 2: Fine-Tuning)

- [x] Qwen2.5-VL-7B fine-tuned to 81.1% (600 samples, embedding similarity)
- [x] Qwen3-VL-8B Q7 baseline at 82% (50 samples)
- [x] Unsloth fine-tuning verified for all 7 Qwen models (RTX 5090)
- [x] Cloud infrastructure built (vast.ai + RunPod)

### Completed (Phase 3: Cloud API Evaluation)

- [x] MedGemma-27B evaluated at 78.81% (709 samples, embedding similarity)
- [x] Gemini batch annotations (62 batches)

### Upcoming

- [ ] Comparative fine-tuning benchmark (top 3 models, same config)
- [ ] Mobile deployment evaluation
- [ ] Web demo with best model
- [ ] Paper writing

## Quick Links

| Phase                   | Status      | Notes                     |
| ----------------------- | ----------- | ------------------------- |
| Phase 1: VLM Benchmarks | Complete    | 50+ models, proxy metrics |
| Phase 2: Fine-Tuning    | Complete    | Qwen2.5-VL-7B at 81.1%    |
| Phase 3: Cloud API      | Complete    | MedGemma-27B at 78.81%    |
| Phase 4: Full Test Set  | In Progress | 1,494 images, GT eval     |
| Phase 5: Web/App Demo   | Pending     | -                         |

## Blockers

- Phase 1 proxy scores (keyword matching) not comparable to Phase 2/3 GT scores (embedding similarity)
- Need standalone inference + GT-eval script for local open-weight models

## Notes

- Full dataset: ~19,000 images, 14 classes, 18,936 annotated
- Hardware: RTX 5090 (24GB VRAM) local, vast.ai/RunPod for cloud
- Best verified GT score: 81.1% (Qwen2.5-VL-7B fine-tuned)
