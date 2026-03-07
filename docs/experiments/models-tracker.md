---
date: 2026-01-27
type: tracker
project: fada-v3
---

## Evaluation Summary

| Phase   | Method               | Sample Size | Ground Truth | Status       |
| ------- | -------------------- | ----------- | ------------ | ------------ |
| Phase 1 | Proxy metrics        | ~250        | No           | Deprecated   |
| Phase 2 | Embedding similarity | 600         | Yes          | Complete     |
| Phase 3 | Cloud API            | 709         | Yes          | Complete     |
| Phase 4 | Full test set        | 1,894       | Yes          | **Complete** |

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

## Phase 4: RCCG H100 Full Test Set Evaluation (Mar 2026)

8x H100 PCIe machines running vLLM, evaluated on 1,894 test images (full test split). 46 models scored.

### Scored Results (zero-shot, 1,894 test images)

Metrics: primary_score = weighted multi-metric average, embed_sim = embedding similarity vs ground truth.

Scoring pipeline v2 (Mar 5): Q5 regex fix, Q2/Q3 keyword expansion, Q1/Q4 abbreviation expansion.
Scoring pipeline v3 (Mar 6): Q1 synonym matching + F2, Q3 transverse synonym, GT spelling fixes. All 46 models rescored.

| Rank | Model                                   | Size | Primary | EmbedSim |
| ---- | --------------------------------------- | ---- | ------- | -------- |
| 1    | Qwen/Qwen3.5-35B-A3B                    | 35B  | 0.3650  | 0.3569   |
| 2    | google/gemma-3-12b-it                   | 12B  | 0.3629  | 0.3641   |
| 3    | OpenGVLab/InternVL3_5-4B                | 4B   | 0.3491  | 0.3946   |
| 4    | Qwen/Qwen3.5-4B                         | 4B   | 0.3460  | 0.3627   |
| 5    | Qwen/Qwen3.5-9B                         | 9B   | 0.3439  | 0.3662   |
| 6    | openbmb/MiniCPM-V-4_5                   | 8B   | 0.3409  | 0.4059   |
| 7    | Qwen/Qwen3-VL-2B-Instruct               | 2B   | 0.3407  | 0.4013   |
| 8    | OpenGVLab/InternVL3_5-8B                | 8B   | 0.3403  | 0.4067   |
| 9    | moonshotai/Kimi-VL-A3B-Instruct         | 3B   | 0.3347  | 0.4057   |
| 10   | mistralai/Mistral-Small-3.1-24B         | 24B  | 0.3292  | 0.4024   |
| 11   | Qwen/Qwen3-VL-8B-Instruct               | 8B   | 0.3288  | 0.3866   |
| 12   | google/gemma-3-4b-it                    | 4B   | 0.3277  | 0.3399   |
| 13   | google/gemma-3-27b-it                   | 27B  | 0.3243  | 0.3686   |
| 14   | Qwen/Qwen2-VL-2B-Instruct               | 2B   | 0.3218  | 0.3720   |
| 15   | JZPeterPan/MedVLM-R1                    | 7B   | 0.3217  | 0.3791   |
| 16   | Qwen/Qwen3-VL-8B-Thinking               | 8B   | 0.3212  | 0.3572   |
| 17   | Qwen/Qwen3.5-0.8B                       | 0.8B | 0.3210  | 0.3771   |
| 18   | Qwen/Qwen3.5-2B                         | 2B   | 0.3207  | 0.3745   |
| 19   | OpenGVLab/InternVL2-8B                  | 8B   | 0.3149  | 0.3768   |
| 20   | moonshotai/Kimi-VL-A3B-Thinking         | 3B   | 0.3144  | 0.3717   |
| 21   | Qwen/Qwen3-VL-4B-Instruct               | 4B   | 0.3136  | 0.3898   |
| 22   | rhymes-ai/Aria                          | 25B  | 0.3051  | 0.3814   |
| 23   | google/medgemma-4b-it                   | 4B   | 0.3035  | 0.4044   |
| 24   | Qwen/Qwen2.5-VL-7B-Instruct             | 7B   | 0.2935  | 0.3828   |
| 25   | OpenGVLab/InternVL2-4B                  | 4B   | 0.2933  | 0.3523   |
| 26   | openbmb/MiniCPM-o-2_6                   | 8B   | 0.2929  | 0.3920   |
| 27   | openbmb/MiniCPM-V-2_6                   | 8B   | 0.2924  | 0.3941   |
| 28   | OpenGVLab/InternVL2-2B                  | 2B   | 0.2900  | 0.3787   |
| 29   | Qwen/Qwen2-VL-7B-Instruct               | 7B   | 0.2897  | 0.3897   |
| 30   | HuggingFaceTB/SmolVLM2-2.2B-Instruct    | 2.2B | 0.2868  | 0.3778   |
| 31   | Qwen/Qwen2.5-Omni-7B                    | 7B   | 0.2800  | 0.4014   |
| 32   | google/gemma-3n-E4B-it                  | 4B   | 0.2787  | 0.3065   |
| 33   | OpenGVLab/InternVL2-1B                  | 1B   | 0.2779  | 0.3606   |
| 34   | HuggingFaceM4/Idefics3-8B-Llama3        | 8B   | 0.2759  | 0.3782   |
| 35   | Qwen/Qwen2.5-VL-3B-Instruct             | 3B   | 0.2639  | 0.3788   |
| 36   | microsoft/Phi-3.5-vision-instruct       | 4B   | 0.2602  | 0.3406   |
| 37   | HuggingFaceTB/SmolVLM-256M-Instruct     | 256M | 0.2441  | 0.3111   |
| 38   | HuggingFaceTB/SmolVLM-500M-Instruct     | 500M | 0.2078  | 0.3541   |
| 39   | lmms-lab/llava-onevision-qwen2-7b-ov-hf | 7B   | 0.1663  | 0.1606   |
| 40   | allenai/Molmo2-8B                       | 8B   | 0.1260  | 0.0155   |
| 41   | OpenGVLab/InternVL2_5-4B                | 4B   | 0.1256  | -0.0088  |
| 42   | mistral-community/pixtral-12b           | 12B  | 0.1256  | -0.0088  |
| 43   | microsoft/Phi-4-multimodal-instruct     | 14B  | 0.1205  | 0.0466   |
| 44   | Qwen/Qwen3.5-27B                        | 27B  | 0.1124  | 0.0330   |
| 45   | NVlabs/Eagle2.5-8B                      | 8B   | 0.0960  | 0.0410   |

Models 40-45 have severe output issues (connection errors, formatting failures, chat template errors).

### Incompatible with vLLM

| Model                                     | Reason                                              |
| ----------------------------------------- | --------------------------------------------------- |
| deepseek-ai/deepseek-vl2-small            | Architecture not supported                          |
| microsoft/llava-med-v1.5-mistral-7b       | Architecture not supported                          |
| lmms-lab/LLaVA-OneVision-1.5-8B-Instruct  | Architecture not supported (LLaVAOneVision1_5)      |
| lmms-lab/LLaVA-OneVision-1.5-4B-Instruct  | Architecture not supported (LLaVAOneVision1_5)      |
| vikhyatk/moondream2                       | Config incompatible (missing hidden_size)           |
| meta-llama/Llama-3.2-11B-Vision-Instruct  | MllamaProcessor missing \_get_num_multimodal_tokens |
| CohereForAI/aya-vision-8b                 | Gated repo, access not approved                     |
| THUDM/glm-4v-9b                           | Engine core initialization failed                   |
| deepseek-ai/deepseek-vl2-tiny             | No model architectures specified                    |
| meta-llama/Llama-4-Scout-17B-16E-Instruct | Gated repo, access not approved                     |
| OpenGVLab/InternVL2_5-4B                  | Chat template error (str/list concat), 100% errors  |
| allenai/Molmo-7B-D-0924                   | Dynamic module import fails (processor error)       |
| Qwen/Qwen2.5-VL-32B-Instruct              | Engine core initialization failed                   |

### Not Yet Available on HuggingFace

- Qwen/Qwen3.5-122B-A10B (too large for single H100)
