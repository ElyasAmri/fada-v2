---
date: 2026-01-27
type: tracker
project: fada-v3
---

## Critical Gap

> **Phase 1 results (MiniCPM 88.9%, etc.) used proxy metrics on ~250 samples WITHOUT ground truth.** These are NOT comparable to Phase 2/3 results. Need to re-evaluate top models on full test set with ground truth.

---

## Priority 1: Ground Truth Evaluation

Re-run top Phase 1 models with embedding similarity scoring:

- [ ] MiniCPM-V-2.6 on 1,494 test images
- [ ] Qwen2-VL-2B on 1,494 test images
- [ ] InternVL2-4B on 1,494 test images
- [ ] Expand Qwen2.5-VL-7B eval from 600 to 1,494

**Why**: Can't compare 88.9% proxy score to 81.1% embedding similarity

---

## Priority 2: Deploy Demo

- [ ] Streamlit web interface with best model
- [ ] Model: Use Qwen2.5-VL-7B (81.1% is REAL score)
- [ ] Location: `models/qwen25vl7b_finetuned/final/`

---

## Priority 3: External Benchmark Validation

- [ ] Evaluate on U2-BENCH (7,241 ultrasound cases, 15 anatomy regions)
- [ ] Compare against 23 SOTA models tested in paper
- [ ] Dataset: [DolphinAI/u2-bench](https://huggingface.co/datasets/DolphinAI/u2-bench)
- [ ] Toolkit: [u2-bench-evalkit](https://github.com/dolphin-sound/u2-bench-evalkit)

**Why**: External validation on standardized benchmark, includes fetal imaging

---

## Priority 4: Address Weak Categories

| Category | MedGemma Score | Issue |
|----------|----------------|-------|
| Trans-cerebellum | 58% | Brain views |
| Trans-thalamic | 57% | Brain views |
| Trans-ventricular | 57% | Brain views |
| NT measurements | 57% | Specialized |

Options:
- [ ] Category-specific fine-tuning
- [ ] More training data for weak categories
- [ ] Ensemble approach
- [ ] Test fetal-specific models (FetalCLIP, EchoVLM)

---

## Completed

- [x] Phase 1: 50+ models zero-shot (proxy metrics)
- [x] Phase 2: Qwen2.5-VL-7B fine-tuned (81.1%, 600 samples)
- [x] Phase 3: MedGemma-27B cloud (78.81%, 709 samples)
- [x] Classification: EfficientNet-B0 (88%, 1,494 samples)
- [x] Infrastructure: vast.ai + RunPod
- [x] Local testing: All 7 Qwen models with Unsloth

---

## Data Status

| Set | Size | Status |
|-----|------|--------|
| Train | 12,014 | Ready |
| Val | 1,494 | Ready |
| Test | 1,494 | **Needs full evaluation** |
| Total | 15,002 | - |

---

## Comparison Matrix (What We Know)

| Model | Method | Samples | Score | Comparable? |
|-------|--------|---------|-------|-------------|
| MiniCPM-V-2.6 | Proxy | ~250 | 88.9% | No |
| Qwen2.5-VL-7B FT | Embedding | 600 | 81.1% | Yes |
| MedGemma-27B | Embedding | 709 | 78.81% | Yes |
| EfficientNet-B0 | Classification | 1,494 | 88% | Different task |
