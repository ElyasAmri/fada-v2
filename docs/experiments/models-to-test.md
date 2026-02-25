---
date: 2026-01-27
type: tracker
project: fada-v3
---

## Fetal Ultrasound Specific Models (HIGH PRIORITY)

| Model | Source | Size | Key Features | Status |
|-------|--------|------|--------------|--------|
| **FetalCLIP** | MBZUAI | - | 210K fetal US images, foundation model | TODO |
| **FetalMind** | ICLR 2026 | - | 12 medical centers, multi-view reasoning | TODO |
| **Sonomate** | Nature Jan 2026 | - | Real-time fetal US, video+audio grounding | TODO |
| **EchoVLM** | arXiv Sep 2025 | MoE | 7 anatomical regions, ultrasound-specific | TODO |
| **LLaVA-Ultra** | - | - | Chinese ultrasound, fine-grained semantics | TODO |
| **AbVLM-Q** | BMC 2025 | - | Abdominal ultrasound quality assessment | TODO |

**Papers:**
- FETAL-GAUGE benchmark (arXiv 2512.22278) - VLM benchmark for fetal US
- "Benchmarking Vision LLMs in fetal ultrasound" (PMC 2025)

---

## General VLMs - Tier 1 (Best Performers)

| Model | Family | Size | VRAM | Status | Notes |
|-------|--------|------|------|--------|-------|
| **MiniCPM-V-2.6** | OpenBMB | 8B | ~16GB | Proxy tested | 88.9% proxy, needs GT eval |
| **Qwen2-VL-2B** | Alibaba | 2B | ~8GB | Proxy tested | 83.3% proxy, efficient |
| **Qwen2-VL-7B** | Alibaba | 7B | ~16GB | Proxy tested | ~75% proxy |
| **Qwen2.5-VL-3B** | Alibaba | 3B | ~10GB | Local tested | Unsloth works |
| **Qwen2.5-VL-7B** | Alibaba | 7B | ~16GB | **Fine-tuned** | 81.1% GT (600 samples) |
| **Qwen3-VL-2B** | Alibaba | 2B | ~8GB | Local tested | Unsloth works |
| **Qwen3-VL-4B** | Alibaba | 4B | ~12GB | Local tested | Unsloth works |
| **Qwen3-VL-8B** | Alibaba | 8B | ~20GB | Local tested | Unsloth works |
| **InternVL2-2B** | OpenGVLab | 2B | ~8GB | Proxy tested | ~80% proxy |
| **InternVL2-4B** | OpenGVLab | 4B | ~12GB | Proxy tested | ~82% proxy |
| **InternVL3-78B** | OpenGVLab | 78B | Cloud | TODO | Top benchmark scores |

---

## General VLMs - Tier 2 (Good Performers)

| Model | Family | Size | VRAM | Status | Notes |
|-------|--------|------|------|--------|-------|
| LLaVA-OneVision | LLaVA | 7B | ~16GB | Proxy tested | ~80% proxy |
| Molmo-7B | AllenAI | 7B | ~16GB | Proxy tested | ~70% proxy |
| PaliGemma2 | Google | 3B | ~10GB | Proxy tested | ~68% proxy |
| Pixtral-12B | Mistral | 12B | ~24GB | TODO | Multi-image, Apache 2.0 |
| Llama-3.2-Vision | Meta | 11B | ~24GB | TODO | Latest Llama vision |
| DeepSeek-VL2 | DeepSeek | 1-4.5B | ~10-16GB | TODO | MoE, strong OCR |
| Gemma-3-Vision | Google | 4B/12B | ~12-24GB | TODO | 128K context |

---

## Medical VLMs

| Model | Focus | Size | Status | Notes |
|-------|-------|------|--------|-------|
| **MedGemma-27B** | General medical | 27B | **Tested** | 78.81% GT (709 samples) |
| MedGemma-4B | General medical | 4B | Proxy tested | ~65% proxy |
| Medical-VLM-24B | General medical | 24B | TODO | 82.9% OpenMed |
| LLaVA-Med | Radiology | 7B | TODO | Medical adaptation |
| CheXagent-8B | Chest X-ray | 8B | Tested | 0% - domain mismatch |

---

## Mobile/Edge VLMs (For Deployment)

| Model | Size | VRAM | Speed | Status |
|-------|------|------|-------|--------|
| SmolVLM-256M | 256M | <1GB | Fast | TODO |
| SmolVLM-500M | 500M | ~1.2GB | Fast | TODO |
| SmolVLM-2.2B | 2.2B | ~5GB | Moderate | TODO |
| Moondream2 | ~2B | ~4GB | 25.6 tok/s | Perf tested |
| MobileVLM-V2-1.7B | 1.7B | Low | 21.5 tok/s | TODO |
| MobileVLM-V2-3B | 3B | Low | Moderate | TODO |
| LFM2-VL-450M | 450M | Low | 2x faster | TODO |
| LFM2-VL-1.6B | 1.6B | Low | Fast | TODO |

---

## API Models

| Model | Provider | Status | Notes |
|-------|----------|--------|-------|
| GPT-4V | OpenAI | TODO | Expensive |
| GPT-4o | OpenAI | TODO | Multimodal |
| Gemini-3-Flash | Google | Available | Fast, cheap |
| Gemini-3-Pro | Google | Available | Best quality |
| Claude-3.5-Sonnet | Anthropic | TODO | Strong reasoning |

---

## Evaluation Priority Order

### Phase 1: Ground Truth Eval (Embedding Similarity)
1. [ ] MiniCPM-V-2.6 (1,494 test images)
2. [ ] Qwen2-VL-2B (1,494 test images)
3. [ ] InternVL2-4B (1,494 test images)
4. [ ] Qwen2.5-VL-7B expand (600 to 1,494)

### Phase 2: Fetal-Specific Models
5. [ ] FetalCLIP
6. [ ] EchoVLM
7. [ ] FetalMind (if available)

### Phase 3: New General VLMs
8. [ ] Qwen3-VL-8B (already local tested, need GT eval)
9. [ ] Llama-3.2-Vision-11B
10. [ ] Pixtral-12B
11. [ ] DeepSeek-VL2

### Phase 4: Mobile Deployment
12. [ ] SmolVLM-2.2B
13. [ ] MobileVLM-V2-3B
14. [ ] Moondream2

---

## External Benchmarks

### U2-BENCH (Recommended)

> First comprehensive benchmark for VLMs on ultrasound understanding (arXiv 2505.17779, ICLR 2026)

| Aspect | Details |
|--------|---------|
| Samples | 7,241 cases |
| Anatomy | 15 regions (includes fetal) |
| Tasks | 8 clinical tasks |
| Models tested | 23 LVLMs |
| Dataset | [HuggingFace](https://huggingface.co/datasets/DolphinAI/u2-bench) |
| Eval toolkit | [GitHub](https://github.com/dolphin-sound/u2-bench-evalkit) |

**8 Tasks:**
1. Disease Diagnosis (DD) - classification
2. View Recognition (VRA) - view classification
3. Lesion Localization (LL) - spatial detection
4. Organ Detection (OD) - anatomical structures
5. Keypoint Detection (KD) - biometry landmarks
6. Clinical Value Estimation (CVE) - regression
7. Report Generation (RG) - text generation
8. Caption Generation (CG) - descriptions

**Use for:** External validation, compare against 23 SOTA models

---

## Key Papers

1. **U2-BENCH** (arXiv 2505.17779) - First ultrasound VLM benchmark, 7,241 cases
2. **FETAL-GAUGE** (arXiv 2512.22278) - Benchmark for VLMs in fetal US
3. **FetalCLIP** (arXiv 2502.14807) - 210K fetal US foundation model
4. **Sonomate** (Nature Jan 2026) - Real-time fetal US understanding
5. **EchoVLM** (arXiv 2509.14977) - Universal ultrasound VLM
6. **FetalMind** (ICLR 2026) - Epistemic-aware fetal US model

---

## Fine-tuning Compatibility

### Unsloth Verified (Local RTX 5090)

| Model | Train Time (2 samples) | Mobile Export |
|-------|------------------------|---------------|
| Qwen2-VL-2B | 44s | Yes (Q4 ~1.5GB) |
| Qwen2-VL-7B | 22s | No (too large) |
| Qwen2.5-VL-3B | 22s | Marginal (Q4 ~2GB) |
| Qwen2.5-VL-7B | 19s | No (too large) |
| Qwen3-VL-2B | 23s | Yes (Q4 ~1.5GB) |
| Qwen3-VL-4B | 24s | Marginal (Q4 ~3GB) |
| Qwen3-VL-8B | 31s | No (too large) |

### Unsloth Likely Compatible (Untested)

| Model | Notes |
|-------|-------|
| Llama-3.2-Vision-11B | Official Unsloth support |
| Pixtral-12B | Mistral architecture supported |

### Other Fine-tuning Methods

| Model | Method |
|-------|--------|
| MiniCPM-V-2.6 | OpenBMB fine-tuning |
| InternVL2-2B/4B | InternVL method |
| LLaVA-OneVision | Standard LLaVA |
| LLaVA-Med | Standard LLaVA |
| PaliGemma2 | Google method |
| Molmo-7B | AllenAI method |
| DeepSeek-VL2 | DeepSeek method |

### Cannot Fine-tune (API Only)

GPT-4V, GPT-4o, Gemini-3-Flash, Gemini-3-Pro, Claude-3.5-Sonnet

---

## Mobile Export Guide

### Best Candidates for Mobile Deployment

| Priority | Model | Q4 Size | Fine-tune | Path |
|----------|-------|---------|-----------|------|
| 1 | Qwen2-VL-2B | ~1.5GB | Unsloth | GGUF export |
| 2 | Qwen3-VL-2B | ~1.5GB | Unsloth | GGUF export |
| 3 | SmolVLM-500M | <1GB | Limited | ONNX native |
| 4 | MobileVLM-V2-1.7B | ~1GB | Standard | TFLite/ONNX |

### Export Pipeline

```
Unsloth fine-tune -> GGUF Q4 -> llama.cpp / MLC-LLM -> Mobile
```

### Mobile-Native Models (No Fine-tuning)

| Model | Size | Format | Notes |
|-------|------|--------|-------|
| SmolVLM-256M | <1GB | ONNX | Purpose-built |
| SmolVLM-500M | ~1GB | ONNX | Purpose-built |
| MobileVLM-V2-1.7B | ~1GB | TFLite | Optimized |
| Moondream2 | ~2GB | GGUF | Fast inference |

---

## Summary Stats

| Category | Count | Tested | TODO |
|----------|-------|--------|------|
| Fetal-specific | 6 | 0 | 6 |
| Tier 1 General | 11 | 7 (proxy) | 4 (GT) |
| Tier 2 General | 7 | 3 (proxy) | 4 |
| Medical | 5 | 2 | 3 |
| Mobile | 8 | 1 (perf) | 7 |
| API | 5 | 0 | 5 |
| **Total** | **42** | **13** | **29** |

### By Fine-tuning Method

| Method | Count |
|--------|-------|
| Unsloth verified | 7 |
| Unsloth likely | 2 |
| Other methods | 10 |
| API only | 5 |
| Zero-shot/unknown | 18 |

### Mobile Viable

| Category | Count |
|----------|-------|
| Unsloth + mobile export | 2-4 |
| Mobile-native models | 8 |
| **Total mobile options** | **10-12** |
