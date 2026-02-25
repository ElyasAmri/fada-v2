---
date: 2026-01-27
type: dashboard
project: fada-v3
---

## Quick Stats

| Metric | Value |
|--------|-------|
| Models Tested | 50+ (Phase 1 proxy) |
| Best Verified Score | Qwen2.5-VL-7B FT (81.1%) |
| Test Set Size | 1,494 images |
| Total Images | 15,002 |
| Categories | 12 |

## Critical Issue

> Phase 1 results (MiniCPM 88.9%) used **proxy metrics** on ~250 samples. NOT comparable to verified results. See [[next-steps]] for action plan.

## Verified Results (Ground Truth)

| Model | Score | Samples | Method |
|-------|-------|---------|--------|
| Qwen2.5-VL-7B FT | 81.1% | 600 | Embedding |
| MedGemma-27B | 78.81% | 709 | Embedding |
| EfficientNet-B0 | 88% | 1,494 | Classification |

## Project Status

```
Phase 1: Proxy Benchmarks     [##########] 100% (needs re-eval)
Phase 2: Ground Truth Eval    [####------] 40%
Phase 3: Cloud API Eval       [########--] 80%
Phase 4: Full Test Set Eval   [----------] 0%
Phase 5: Web Interface        [----------] 0%
```

## Links

- [[models-tracker|Models Tracker]]
- [[results-summary|Results Summary]]
- [[next-steps|Next Steps]]
- [[All Models Tested|All 50+ Models]]
- [[timesheets/january_2026|January 2026 Timesheet]]

## Next Action

**Re-evaluate top Phase 1 models with ground truth** on full 1,494 test set to get comparable scores.
