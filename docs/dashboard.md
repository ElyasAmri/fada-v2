---
date: 2026-01-27
type: dashboard
project: fada-v3
---

## Quick Stats

| Metric              | Value                    |
| ------------------- | ------------------------ |
| Models Tested       | 46 (GT scored)           |
| Best Verified Score | Qwen2.5-VL-7B FT (81.1%) |
| Test Set Size       | 1,894 images             |
| Total Images        | 19,019                   |
| Categories          | 14                       |

## Verified Results (Ground Truth)

| Model            | Score  | Samples | Method         |
| ---------------- | ------ | ------- | -------------- |
| Qwen2.5-VL-7B FT | 81.1%  | 600     | Embedding      |
| MedGemma-27B     | 78.81% | 709     | Embedding      |
| EfficientNet-B0  | 88%    | 1,494   | Classification |
| Qwen3.5-35B-A3B  | 36.5%  | 1,894   | GT primary     |
| gemma-3-12b-it   | 36.3%  | 1,894   | GT primary     |
| InternVL3_5-4B   | 34.9%  | 1,894   | GT primary     |

## Project Status

```
Phase 1: VLM Benchmarking     [##########] 100%
Phase 2: Ground Truth Eval    [##########] 100%
Phase 3: Cloud API Eval       [##########] 100%
Phase 4: Full Test Set Eval   [##########] 100% (46 models, 1,894 images)
Phase 5: Web Interface        [----------] 0%
```

## Links

- [[models-tracker|Models Tracker]]
- [[results-summary|Results Summary]]
- [[next-steps|Next Steps]]
- [[All Models Tested|All 50+ Models]]
- [[timesheets/january_2026|January 2026 Timesheet]]

## Next Action

**Fine-tune top-performing models** and evaluate on full 1,894 test set.
