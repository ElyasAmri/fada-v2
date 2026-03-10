# VLM Evaluation Analysis - RCCG Benchmark

**Date:** 2026-03-05
**Analyst:** Scientist agent (oh-my-claudecode)

---

## [OBJECTIVE]

Comprehensive analysis of 17 Vision Language Models evaluated on 1,894 fetal ultrasound test images
across 8 clinical questions and 14 anatomical categories. Primary goal: identify performance patterns,
category difficulty, qualitative failure modes, and whether model size or medical specialization predict
performance.

---

## [DATA]

- **Test set:** 1,894 images, 8 questions each = 15,152 Q/A pairs per model
- **Matched for scoring:** 15,080 pairs per model (99.5% match rate); 72 unmatched across all models (consistent annotation gap, not a model-specific issue)
- **Models evaluated:** 17 (score files in `scores/`); 1 additional (Qwen3-VL-8B-Thinking) has predictions but no score file yet
- **Metrics:** `primary_score` (weighted multi-metric average across 8 questions) and `embedding_similarity` (sentence-transformer cosine similarity vs ground truth)
- **Metric types per question:**
  - Q1: Anatomical Structures -- set_f1
  - Q2: Fetal Orientation -- relaxed_accuracy
  - Q3: Imaging Plane -- relaxed_accuracy
  - Q4: Biometric Measurements -- keyword_f1
  - Q5: Gestational Age -- exact_bin_match
  - Q6: Image Quality -- exact_tier_match
  - Q7: Normality Assessment -- exact_match_with_binary
  - Q8: Clinical Recommendations -- bertscore_f1

---

## 1. Overall Model Rankings

[FINDING] Performance spans a wide range: best model (Gemma-3-12B) scores 0.3039 primary vs worst valid model (Qwen2.5-VL-3B) at 0.2380 among models without infrastructure failures. Phi-4 and LLaVA-OV are disqualified by systematic API errors (see Section 5).
[STAT:n] n=15,080 scored pairs per model
[STAT:effect_size] Score range among valid models [0.2380, 0.3039]; mean=0.2686; std=0.0178

| Rank | Model | Primary Score | Embedding Sim | Notes |
|------|-------|--------------|---------------|-------|
| 1 | google_gemma-3-12b-it | **0.3039** | 0.3642 | |
| 2 | OpenGVLab_InternVL3_5-4B | 0.2935 | 0.3947 | |
| 3 | mistralai_Mistral-Small-3.1-24B-Instruct-2503 | 0.2927 | 0.4024 | |
| 4 | OpenGVLab_InternVL3_5-8B | 0.2903 | 0.4066 | |
| 5 | openbmb_MiniCPM-V-4_5 | 0.2887 | 0.4058 | |
| 6 | JZPeterPan_MedVLM-R1 | 0.2803 | 0.3792 | Medical-specific |
| 7 | Qwen_Qwen3-VL-8B-Instruct | 0.2802 | 0.3866 | |
| 8 | google_gemma-3-4b-it | 0.2747 | 0.3401 | |
| 9 | Qwen_Qwen3-VL-4B-Instruct | 0.2674 | 0.3897 | |
| 10 | Qwen_Qwen2.5-VL-7B-Instruct | 0.2593 | 0.3828 | |
| 11 | openbmb_MiniCPM-V-2_6 | 0.2525 | 0.3942 | |
| 12 | google_medgemma-4b-it | 0.2502 | 0.4044 | Medical-specific |
| 13 | InternVL2-4B | 0.2483 | 0.3522 | Older architecture |
| 14 | HuggingFaceTB_SmolVLM2-2.2B-Instruct | 0.2456 | 0.3779 | Smallest model |
| 15 | Qwen_Qwen2.5-VL-3B-Instruct | 0.2380 | 0.3788 | |
| 16 | llava-hf_llava-onevision-qwen2-7b-ov-hf | 0.1530 | 0.1606 | **64.8% API errors** |
| 17 | microsoft_Phi-4-multimodal-instruct | 0.1200 | 0.0466 | **95.6% API errors** |

[STAT:effect_size] Pearson r between embedding_similarity and primary_score across all 17 models: r=0.91 (very strong). Both metrics rank models consistently.

---

## 2. Per-Question Difficulty

[FINDING] Question difficulty varies enormously. Q8 (Clinical Recommendations, BERTScore) and Q6 (Image Quality, exact tier match) are solved by all models, while Q5 (Gestational Age, exact bin match) is impossible -- every model scored exactly 0.0000.
[STAT:n] n=1,885 samples per question per model (17 models)

| Question | Metric | Mean (17 models) | Min | Max | Difficulty |
|----------|--------|-----------------|-----|-----|------------|
| Q8: Clinical Recommendations | bertscore_f1 | **0.7781** | 0.7589 | 0.8002 | Easiest |
| Q6: Image Quality | exact_tier_match | 0.6651 | 0.0411 | 0.8491 | Easy |
| Q7: Normality Assessment | exact_match_with_binary | 0.2587 | 0.0032 | 0.4912 | Moderate |
| Q4: Biometric Measurements | keyword_f1 | 0.1392 | 0.0913 | 0.1835 | Hard |
| Q1: Anatomical Structures | set_f1 | 0.0999 | 0.0067 | 0.1464 | Hard |
| Q3: Imaging Plane | relaxed_accuracy | 0.0644 | 0.0000 | 0.1416 | Very Hard |
| Q2: Fetal Orientation | relaxed_accuracy | 0.0363 | 0.0000 | 0.1255 | Very Hard |
| Q5: Gestational Age | exact_bin_match | **0.0000** | 0.0000 | 0.0000 | Impossible |

[FINDING] Q5 (Gestational Age, exact_bin_match) scored exactly 0.0 for all 17 models without exception. This is not a model performance finding -- it indicates the metric is broken, the ground truth bins are not inferable from image content, or there is a systematic encoding mismatch between model outputs and expected bins. This requires investigation before the metric can be considered valid.
[STAT:n] n=17 models, all 0.0000; variance=0

[FINDING] Q8 (Clinical Recommendations, BERTScore) has the narrowest variance across models (range 0.7589-0.8002, spread=0.041), while Q2 has the widest relative spread (CV=121%). Q8 rewards any fluent clinical text regardless of factual accuracy and has minimal discriminative power.
[STAT:effect_size] Q8 CV=1.8%; Q2 CV=121%; Q3 CV=72%

[FINDING] Q7 (Normality Assessment) is systematically biased toward predicting "normal" across all models. The top-5 models all show a "predicts normal" rate of 98.3-98.7% of samples. All models have near-zero sensitivity (range: 0.067-0.423) indicating a class-imbalance failure -- the test set contains very few anomalous images and models learn to default to "normal".
[STAT:n] n=1,717-1,885 per model for Q7

Top-5 model Q7 confusion matrix summary:

| Model | Sensitivity | Specificity | TP | FP | FN |
|-------|------------|-------------|----|----|-----|
| gemma-3-12b-it | 0.200 | 0.744 | 6 | 432 | 24 |
| InternVL3.5-4B | 0.217 | 0.810 | 5 | 327 | 18 |
| Mistral-24B | 0.423 | 0.732 | 11 | 429 | 15 |
| InternVL3.5-8B | 0.067 | 0.954 | 2 | 85 | 28 |
| MiniCPM-V-4.5 | 0.333 | 0.788 | 8 | 310 | 16 |

---

## 3. Per-Category (Anatomical) Performance

[FINDING] Femur and CRL-View are the easiest anatomical categories (mean primary ~0.29) while Aorta and Non_standard_NT are hardest (mean ~0.19-0.20) -- a ~50% performance gap between best and worst category across all models.
[STAT:n] n=17 models per category

| Category | Mean Primary | Min | Max | Difficulty |
|----------|-------------|-----|-----|------------|
| Femur | **0.2927** | 0.0992 | 0.3710 | Easiest |
| CRL-View | 0.2902 | 0.0981 | 0.3625 | Easy |
| Trans-thalamic | 0.2833 | 0.1003 | 0.3644 | Easy |
| Trans-ventricular | 0.2807 | 0.0996 | 0.3646 | Easy |
| NT-View | 0.2655 | 0.1011 | 0.3432 | Moderate |
| Abdomen | 0.2586 | 0.1173 | 0.3336 | Moderate |
| Cervical | 0.2566 | 0.1063 | 0.3474 | Moderate |
| Public_Symphysis_fetal_head | 0.2488 | 0.1052 | 0.3292 | Moderate |
| Thorax | 0.2481 | 0.1976 | 0.3535 | Moderate |
| Trans-cerebellum | 0.2407 | 0.1019 | 0.3033 | Hard |
| Standard_NT | 0.2365 | 0.0999 | 0.3161 | Hard |
| Cervix | 0.2337 | 0.0993 | 0.2891 | Hard |
| Non_standard_NT | 0.2004 | 0.1013 | 0.2311 | Very Hard |
| Aorta | **0.1941** | 0.0992 | 0.2469 | Hardest |

[FINDING] Thorax has the highest minimum score of any category (0.1976), meaning even weak models can handle it. This likely reflects that thoracic views are visually distinctive. In contrast, Aorta and Cervix have lower minima, indicating high model-to-model variance suggesting these views require fine-grained anatomical reasoning.

[FINDING] Non_standard_NT is the most compressed category -- its maximum score (0.2311) is below the overall mean (0.2552). No model handles non-standard nuchal translucency views well. This is clinically significant since NT measurement is one of the primary markers for chromosomal anomalies in first-trimester screening.
[STAT:effect_size] Non_standard_NT max (0.2311) < overall cross-model mean (0.2552)

---

## 4. Model Size and Medical Specialization Patterns

### 4a. Does Model Size Predict Performance?

[FINDING] Model size is a weak positive predictor of performance (Pearson r=0.33, n=13 models with parseable parameter counts). Within-family comparisons are inconsistent: 3 of 5 families show larger = better, 1 shows larger = worse (InternVL3.5), and 1 is approximately equal (InternVL3.5 delta=-0.003).
[STAT:effect_size] Pearson r=0.33 (weak positive); 4 out of 5 within-family comparisons show larger is better or equal
[STAT:n] n=13 models with parseable sizes

| Family | Small Model | Score | Large Model | Score | Delta |
|--------|------------|-------|------------|-------|-------|
| Qwen2.5-VL | 3B: 0.2380 | | 7B: 0.2593 | | +0.021 |
| Qwen3-VL | 4B: 0.2674 | | 8B: 0.2802 | | +0.013 |
| InternVL3.5 | 4B: **0.2935** | | 8B: 0.2903 | | -0.003 |
| Gemma-3 | 4B: 0.2747 | | 12B: **0.3039** | | +0.029 |
| MiniCPM-V | 2.6B: 0.2525 | | 4.5B: 0.2887 | | +0.036 |

[FINDING] InternVL3.5-4B is an outlier: it ranks 2nd overall at 0.2935 despite being a 4B model, outperforming both InternVL3.5-8B and the 24B Mistral-Small. The 24B Mistral-Small ranks 3rd (0.2927), only narrowly ahead of InternVL3.5-8B at 4th (0.2903). Architecture and instruction tuning appear more predictive than raw parameter count.
[STAT:effect_size] InternVL3.5-4B vs 24B Mistral-Small: delta=-0.0008 (essentially equal despite 6x parameter difference)

### 4b. Do Medical-Specific Models Outperform General Models?

[FINDING] Medical-specific fine-tuning does not provide a systematic advantage on this benchmark. MedGemma-4B (0.2502) underperforms its general-purpose sibling Gemma-3-4B (0.2747) by 0.0245 on identical architecture -- medical fine-tuning causes a regression. MedVLM-R1 (0.2803) achieves mid-table performance (6th of 15 valid models), but its base architecture is unknown, preventing a controlled comparison.
[STAT:effect_size] MedGemma-4B vs Gemma-3-4B delta: -0.0245 (medical FT hurts performance)
[STAT:n] n=15,080 pairs per model

| Model | Type | Primary Score | Comparison |
|-------|------|--------------|------------|
| google_gemma-3-4b-it | General (baseline) | 0.2747 | Reference |
| google_medgemma-4b-it | Medical FT (same arch) | 0.2502 | -0.0245 vs baseline |
| JZPeterPan_MedVLM-R1 | Medical (unknown base) | 0.2803 | Uncontrolled |

Mean score of 2 medical models: 0.2653 vs mean of 15 general models: 0.2539. The medical mean is higher, but this is driven by MedVLM-R1 being an 8B-class model; the only controlled comparison (MedGemma vs Gemma-3-4B) shows medical fine-tuning hurts.

---

## 5. Infrastructure Failures: LLaVA-OV and Phi-4

[FINDING] Two models have results that are invalid for benchmarking due to systematic infrastructure failures. Their scores must not be compared to the remaining 15 models.

**microsoft_Phi-4-multimodal-instruct (rank 17, primary=0.1200):**
- 95.6% of 15,152 predictions: `ERROR: vLLM API error: Connection error.`
- 0.2% additional: `No response generated.`
- Valid response rate: ~4.2%
- Root cause: vLLM server connection dropped during inference. Checkpoint file (18.1 MB, smallest of all) confirms minimal work completed.
[STAT:n] 14,483/15,152 error responses (95.6%)

**llava-hf_llava-onevision-qwen2-7b-ov-hf (rank 16, primary=0.1530):**
- 64.8% of 15,152 predictions: `ERROR: vLLM API error: Error code: 400`
- Error: `"The decoder prompt (length 4774) is longer than the maximum model length of 4096"`
- Root cause: LLaVA-OV was deployed with `max_model_len=4096`, insufficient for prompts that include image tokens plus multi-question system prompts. Questions Q5-Q8 (longer system prompts) fail for virtually all images; earlier questions partially succeed.
- Valid response rate: ~35.2%
[STAT:n] 9,816/15,152 error responses (64.8%)

Both models should be re-run with corrected vLLM configuration (`max_model_len` >= 8192 for LLaVA-OV; stable server for Phi-4) before any conclusions about their capabilities.

---

## 6. Thinking Model Analysis: Qwen3-VL-8B-Thinking

[FINDING] The Qwen3-VL-8B-Thinking variant generates responses approximately 2x longer than the standard Qwen3-VL-8B-Instruct (mean 4,009 chars vs 2,032 chars). This is consistent with chain-of-thought reasoning tokens being included in output text rather than stripped. The larger checkpoint (131.9 MB vs 74.5 MB, ratio 1.77x) reflects this additional stored text.
[STAT:n] n=15,152 predictions each
[STAT:effect_size] Mean response length ratio: 1.97x; checkpoint size ratio: 1.77x

| Model | Mean Response Length | Min | Max | Checkpoint Size |
|-------|---------------------|-----|-----|-----------------|
| Qwen3-VL-8B-Thinking | 4,009 chars | 1,131 | 5,337 | 131.9 MB |
| Qwen3-VL-8B-Instruct | 2,032 chars | 515 | 5,222 | 74.5 MB |
| gemma-3-12b-it (best) | 2,397 chars | 541 | 5,134 | 84.6 MB |
| Phi-4-multimodal (worst) | 110 chars | 22 | 5,240 | 18.1 MB |

The Thinking model responses include visible chain-of-thought preamble ("Wait, the image shows...", "Let me check...", "Starting with the main structure...") as the first portion of each response, with the structured answer following. The Thinking checkpoint (`checkpoint_vllm_Qwen_Qwen3-VL-8B-Thinking.json`) confirms all 1,894 images are complete. No score file has been generated yet.

[LIMITATION] The Thinking model is unscored. Whether the 2x response length produces better primary_score is unknown. Chain-of-thought text prepended to the final answer may hurt structured metrics (set_f1 for Q1, exact_tier_match for Q6) if the scoring pipeline extracts the full response text rather than isolating the final answer.

---

## 7. Qualitative Comparison: Best vs Worst Models

**Image:** `Abdomen/Abdomen_006.png`

### Q1 - Anatomical Structures Identification

**gemma-3-12b-it (best):** Provides specific, image-grounded analysis with structured anatomical observations: "the fetal head appearing as a rounded, echogenic structure", "outline of the skull". Uses correct clinical terminology and acknowledges resolution limitations. Responds to what is actually visible in the image.

**microsoft_Phi-4 (worst):** Responds "The image is not visible to me, but I can guide you on how to identify anatomical structures in a typical fetal ultrasound image." This is a hallucinated inability to process the image, followed by generic educational content. Reflects the 95.6% connection error context where the model fell back to text-only generic responses.

### Q6 - Image Quality Assessment

**gemma-3-12b-it:** Rates image as "acceptable with limitations", identifies acoustic shadowing, assesses clarity as moderate. Uses proper ultrasound quality vocabulary.

**Phi-4 (valid 4% responses):** Describes image as "high-quality, grayscale, 2D ultrasound with a resolution of 20 frames per second" -- fps is a temporal metric irrelevant to still images. Generic template response not grounded in the actual image.

### Q8 - Clinical Recommendations

**gemma-3-12b-it:** References specific image metadata visible in the ultrasound (gestational age 29.2 weeks, THI 4.4 MHz transducer), providing context-appropriate recommendations grounded in the image content.

**Phi-4 (valid responses):** Provides generic multi-point recommendation list ("Use of 9 dB gain and 4.4 MHz transducer frequency seems appropriate") that could apply to any fetal ultrasound without specific grounding.

Key difference: Gemma-3-12B demonstrates genuine image grounding where answers change with image content. The valid Phi-4 responses are image-agnostic templates.

---

## 8. Summary of Key Findings

[FINDING] The top-performing cluster (ranks 1-7) spans primary scores 0.2803-0.3039, a gap of only 0.024 (8% relative). This tight cluster suggests models in the 4B-24B range have converged on a performance ceiling imposed by Q5 (zero for all), Q2, and Q3 (both below 0.13 mean).
[STAT:effect_size] Range within top-7: 0.024; relative gap: 8% of cluster mean (0.2906)
[STAT:n] n=15,080 per model

[FINDING] Primary_score is heavily distorted by two degenerate questions: Q5 (all zeros) and Q8 (near-constant ~0.78 across all models). Removing these two questions would likely reorder the leaderboard substantially and better reflect clinically meaningful capabilities.
[STAT:effect_size] Q8 spread across models: 0.041 (5.4% relative); Q5 variance: 0 across all models

[FINDING] The 72 unmatched annotation pairs are consistent across all 17 models, confirming this is a dataset issue (missing or misnamed images in the annotation file) rather than a model-specific problem. These 72 pairs represent 0.5% of the test set and are unlikely to materially affect rankings.
[STAT:n] 72 unmatched / 15,152 total = 0.5% across all 17 models

---

## [LIMITATION]

1. **Q5 zero-score anomaly:** All 17 models score exactly 0.0 on Gestational Age (exact_bin_match). Requires investigation before this metric can be considered valid.

2. **Two disqualified models:** LLaVA-OV and Phi-4 have infrastructure failure rates of 64.8% and 95.6%. Their scores do not reflect model capability. Both should be re-run.

3. **Medical model sample:** Only 2 medical-specific models tested. The MedGemma vs Gemma-3-4B comparison is the only controlled experiment. Broader conclusions about medical VLMs are not supported.

4. **Class imbalance in Q7:** Near-zero sensitivity across all models indicates normality assessment is measuring "predict normal always" rather than genuine anomaly detection. The metric requires a balanced test set or a different evaluation approach (e.g., stratified sampling, F1 over anomaly class).

5. **Thinking model unscored:** Qwen3-VL-8B-Thinking predictions are complete but unscored. Whether chain-of-thought reasoning improves or degrades performance on structured metrics is unknown.

6. **Primary score metric design:** Q8 (BERTScore) and Q6 (Image Quality tier) together account for disproportionate weight in the primary_score average while having the lowest discriminative power. The metric weighting scheme should be revisited.

7. **Comparison to prior fine-tuned results:** The documented best score of 81.1% (Qwen2.5-VL-7B fine-tuned) used embedding similarity only on 600 samples with a different evaluation setup. It is not comparable to the primary_score metric in this analysis.

8. **72 unmatched annotations:** Consistent 0.5% blind spot across all models. Root cause (likely filename mismatch in annotation file) should be fixed to ensure complete test coverage.

---

## Files Referenced

- Scores: `experiments/rccg/results/scores/*.json` (17 files)
- Predictions: `experiments/rccg/results/predictions/*.jsonl` (18 files, including Thinking)
- Checkpoints: `experiments/rccg/results/checkpoint_vllm_*.json` (18 files)
- Ground truth: `data/Fetal Ultrasound Annotations Normalized.xlsx`
