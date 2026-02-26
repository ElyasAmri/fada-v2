---
date: 2026-01-27
type: tracker
project: fada-v3
---

## Evaluation Summary

| Phase   | Method               | Sample Size | Ground Truth | Status   |
| ------- | -------------------- | ----------- | ------------ | -------- |
| Phase 1 | Proxy metrics        | ~250        | No           | Complete |
| Phase 2 | Embedding similarity | 600         | Yes          | Complete |
| Phase 3 | Cloud API            | 709         | Yes          | Complete |
| Phase 4 | Full test set        | 1,494       | Yes          | **TODO** |

---

## Phase 1: Zero-Shot Proxy Metrics (Oct 2025)

> **WARNING**: These used proxy evaluation WITHOUT ground truth annotations. Sample size ~250. Not directly comparable to Phase 2/3 results.

| Rank | Model           | Proxy Score | Size | Notes            |
| ---- | --------------- | ----------- | ---- | ---------------- |
| 1    | MiniCPM-V-2.6   | 88.9%       | 8B   | Best proxy score |
| 2    | Qwen2-VL-2B     | 83.3%       | 2B   | Efficient        |
| 3    | InternVL2-4B    | ~82%        | 4B   | -                |
| 4    | InternVL2-2B    | ~80%        | 2B   | -                |
| 5    | LLaVA-OneVision | ~80%        | 7B   | -                |
| 6    | Qwen2-VL-7B     | ~75%        | 7B   | -                |
| 7    | Molmo-7B        | ~70%        | 7B   | -                |
| 8    | PaliGemma2      | ~68%        | 3B   | -                |
| 9    | MedGemma-4B     | ~65%        | 4B   | -                |
| 10   | BLIP-2          | ~55%        | 3B   | Baseline         |
| -    | CheXagent-8B    | 0%          | 8B   | Chest X-ray only |

**50+ models tested total** - see `docs/experiments/vlm/all_models_tested.md` for Phase 1 history, `docs/experiments/models-to-test.md` for current model list and priority order

---

## Phase 2: Fine-Tuned with Ground Truth (Jan 2026)

| Model             | Task         | Score     | Samples | Method               |
| ----------------- | ------------ | --------- | ------- | -------------------- |
| **Qwen2.5-VL-7B** | VQA Q1-Q8    | **81.1%** | 600     | Embedding similarity |
| Qwen3-VL-8B       | Q7 Normality | 82%       | 50      | Accuracy             |
| EfficientNet-B0   | 12-class     | 88%       | 1,494   | Classification       |

---

## Phase 3: Cloud API Evaluation (Dec 2025)

| Model            | Provider  | Score      | Samples | Notes      |
| ---------------- | --------- | ---------- | ------- | ---------- |
| **MedGemma-27B** | Vertex AI | **78.81%** | 709     | Full Q1-Q8 |

---

## Local GPU Testing (RTX 5090, 24GB)

All 7 Qwen models verified working with Unsloth:

| Model         | Unsloth | 4-bit | Train Time (2 samples) |
| ------------- | ------- | ----- | ---------------------- |
| qwen2-vl-2b   | Yes     | Yes   | 44s                    |
| qwen2.5-vl-3b | Yes     | Yes   | 22s                    |
| qwen2-vl-7b   | Yes     | Yes   | 22s                    |
| qwen2.5-vl-7b | Yes     | Yes   | 19s                    |
| qwen3-vl-2b   | Yes     | Yes   | 23s                    |
| qwen3-vl-4b   | Yes     | Yes   | 24s                    |
| qwen3-vl-8b   | Yes     | Yes   | 31s                    |

---

## TODO: Phase 4 Evaluation

See `docs/experiments/models-to-test.md` for the full prioritized list (54 models).

Top priorities for GT evaluation on 1,494 test images:

1. [ ] Qwen3-VL-8B (local tested, needs GT eval)
2. [ ] Qwen3.5-35B-A3B (3B active, current-gen)
3. [ ] InternVL3.5-8B (current-gen, Cascade RL)
4. [ ] LLaVA-OneVision-1.5-8B (beats Qwen2.5-VL-7B on 18/27 benchmarks)
5. [ ] MiniCPM-V-2.6 (88.9% proxy, needs GT re-eval)
6. [ ] Qwen2.5-VL-7B (expand from 600 to 1,494 samples)
