---
date: 2026-01-27
type: results
project: fada-v3
---

## Per-Category Performance

### Qwen2.5-VL-7B Fine-Tuned (Best)

| Category | Similarity | Status |
|----------|------------|--------|
| Abdomen | 90.9% | Excellent |
| Standard_NT | 82.9% | Good |
| Non_standard_NT | 82.6% | Good |
| Femur | 81.4% | Good |
| Trans-thalamic | 77.8% | Fair |
| Cervix | 77.5% | Fair |
| Thorax | 77.5% | Fair |
| **Mean** | **81.1%** | - |

### MedGemma-27B (Cloud API)

| Category | Score | Status |
|----------|-------|--------|
| Abdomen | 87.87% | Excellent |
| Femur | 87.59% | Excellent |
| Cervical | 78.21% | Good |
| Cervix | 71.01% | Fair |
| Trans-cerebellum | 58% | Poor |
| Trans-thalamic | 57% | Poor |
| Trans-ventricular | 57% | Poor |
| NT measurements | 57% | Poor |
| **Overall** | **78.81%** | - |

## Classification Results (EfficientNet-B0)

| Metric | Value |
|--------|-------|
| Test Accuracy | 88% |
| Balanced Accuracy | 85.7% |
| Categories | 12 |
| Epochs | 26 |

## Key Findings

1. **Abdomen/Femur easiest** - All models score 80%+ on these categories
2. **Brain views challenging** - Trans-* categories consistently ~57-78%
3. **NT measurements difficult** - ~57% across models
4. **Fine-tuning helps** - +6% improvement over zero-shot for Qwen

## Data Summary

| Category | Images | Annotations |
|----------|--------|-------------|
| Abdomen | 100+ | Complete |
| Aorta | 50+ | Complete |
| Cervical | 50+ | Complete |
| Cervix | 50+ | Complete |
| Femur | 50+ | Complete |
| Non_standard_NT | 50+ | Complete |
| Public_Symphysis | 50+ | Complete |
| Standard_NT | 50+ | Complete |
| Thorax | 50+ | Complete |
| Trans-cerebellum | 50+ | Complete |
| Trans-thalamic | 50+ | Complete |
| Trans-ventricular | 50+ | Complete |
| **Total** | **~750+** | 8 Q per image |
