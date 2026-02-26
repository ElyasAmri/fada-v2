# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FADA (Fetal Anomaly Detection Algorithm) is a research prototype for fetal ultrasound image analysis using Vision Language Models (VLMs). The project benchmarks, fine-tunes, and evaluates VLMs for automated ultrasound interpretation via 8 clinical questions per image. NOT for clinical use.

### Current Focus

- **VLM Benchmarking**: 54 models tracked across 6 categories (current-gen: Qwen3.5, InternVL3.5, LLaVA-OV-1.5, etc.)
- **Best verified score**: Qwen2.5-VL-7B fine-tuned at 81.1% (embedding similarity, 600 samples)
- **Full dataset**: ~19,000 images, 14 anatomical classes, 18,936 annotated
- **External benchmark**: U2-BENCH (ICLR 2026) -- 17/21 leaderboard models in our test list

### Key Project Files to Review First

1. `docs/project/project.md` - Project description and phases
2. `docs/project/Tasks.md` - Current task status
3. `docs/experiments/models-to-test.md` - Model list, priority order, fine-tuning compatibility
4. `docs/experiments/models-tracker.md` - Model evaluation status
5. `docs/experiments/results-summary.md` - Per-category performance
6. `docs/project/next-steps.md` - Prioritized action plan

## Key Commands

### Environment Setup (Windows)

```bash
# IMPORTANT: Use forward slashes in bash on Windows
./venv/Scripts/python.exe  # Correct
venv\Scripts\python.exe    # WRONG - will fail in bash

# Activate virtual environment
source venv/Scripts/activate  # Git Bash/WSL

# Install all dependencies (includes PyTorch with CUDA 12.8)
pip install -r requirements.txt
```

### VLM Experiments

```bash
# Run API model evaluation (Gemini, GPT-4o, vLLM backends)
./venv/Scripts/python.exe experiments/api_models/test_api_vlm.py

# Fine-tune with Unsloth (local, RTX 5090)
./venv/Scripts/python.exe experiments/unsloth_vlm/train_qwen3vl.py

# Fine-tune with LoRA (cloud, vast.ai/RunPod)
./venv/Scripts/python.exe experiments/fine_tuning/train_qwen3vl_lora.py

# Track experiments with MLflow
mlflow ui --host 0.0.0.0 --port 5000
```

### Classification Model (Legacy)

```bash
# Train EfficientNet-B0 classification model
./venv/Scripts/python.exe src/training/train_classification.py
```

### Web Interface

```bash
# Run Streamlit prototype (NOT for clinical use)
streamlit run web/app.py --server.port 8501
```

## Project Architecture

### VLM Evaluation Pipeline

```
Dataset (19K images, 8 questions each)
+-- API Models (Gemini, GPT-4o via experiments/api_models/)
+-- Local Models (Qwen3-VL, InternVL3.5, MiniCPM via src/inference/)
+-- Fine-tuning (Unsloth local, LoRA cloud)
+-- Evaluation (embedding similarity vs ground truth)
```

### Source Code (src/)

```
src/
+-- inference/           # VLM inference interfaces
|   +-- api/             # Gemini, GPT-4o, Grok backends
|   +-- local/           # Qwen, InternVL, MiniCPM, Moondream
+-- data/                # Dataset loading, augmentation, VLM dataset
+-- models/              # Classifier, VQA model
+-- chatbot/             # Response generation
+-- training/            # Classification training
+-- utils/               # MLflow, metrics, visualization
```

### Key Design Decisions

- **Primary approach**: VLM benchmarking and fine-tuning (not classification)
- **Evaluation metric**: Embedding similarity (sentence-transformers) vs ground truth
- **Fine-tuning**: LoRA adapters (r=16, alpha=32, all linear targets)
- **Local training**: Unsloth on RTX 5090 (verified for 7 Qwen models)
- **Cloud training**: vast.ai / RunPod (RTX 3090/4090 at ~$0.40/h)
- **Web Framework**: Streamlit for prototype

## Data Structure

```
data/Fetal Ultrasound/
+-- Abodomen/            # 2,424 images (note: typo in folder name is intentional)
+-- Aorta/               # 1,308 images
+-- CRL-View/            # 1,989 images
+-- Cervical/            # 500 images
+-- Cervix/              # 1,626 images
+-- Femur/               # 1,165 images
+-- NT-View/             # 2,028 images
+-- Non_standard_NT/     # 487 images
+-- Public_Symphysis_fetal_head/  # 1,358 images
+-- Standard_NT/         # 1,508 images
+-- Thorax/              # 1,793 images
+-- Trans-cerebellum/    # 684 images
+-- Trans-thalamic/      # 1,565 images
+-- Trans-ventricular/   # 584 images
+-- annotations.xlsx     # Per-class annotations
data/Fetal Ultrasound Annotations Final.xlsx  # Master annotations (18,936 rows)
data/vlm_training/       # JSONL training files for VLM fine-tuning
data/dataset_splits.json # Train/val/test split (12,014 / 1,494 / 1,494)
```

## Critical Context

- **Hardware**: RTX 5090 (24GB VRAM) local, cloud GPUs for benchmarking
- **Accuracy**: Best verified GT score is 81.1% (Qwen2.5-VL-7B fine-tuned)
- **Key caveat**: Phase 1 proxy scores (keyword matching) are NOT comparable to Phase 2/3 GT scores (embedding similarity)
- **Documentation**: Every decision documented for research paper

## Common Issues and Solutions

### Windows Path Issues

- Always use forward slashes in bash: `./venv/Scripts/python.exe`
- Git Bash requires Unix-style paths
- Use `source` not `.` for activation in Git Bash

### Cross-Platform Image Paths

- Cloud scripts may fallback to dummy images when Windows paths detected
- See GitHub issue #6 for the RunPod image path bug

## Docs Structure

```
docs/
+-- dashboard.md                  # Project overview hub
+-- project/                      # Project management
|   +-- project.md, spec.md, Tasks.md, next-steps.md, Narrative.md
+-- experiments/                  # Experiment docs and tracking
|   +-- models-to-test.md         # 54 models, priority order, U2-BENCH cross-ref
|   +-- models-tracker.md, results-summary.md
|   +-- vlm/                     # VLM testing results
|   +-- Evaluation Methodology.md, Fine-Tuning Approach.md
|   +-- Vastai CLI Implementation.md
+-- papers/                       # Research paper artifacts
|   +-- bibliography.md, literature_review.md, Paper Outline.md
+-- reports/                      # LaTeX report
+-- timesheets/                   # Private (gitignored)
```

## API Model Defaults

- **Gemini API**: Always use `gemini-3-flash-preview` as the default model (NOT gemini-2.5-flash)
- **Batch annotations**: Use Gemini 3 Flash for all new annotation jobs
- **OpenAI API**: Use GPT-4o for vision tasks

## Critical Instructions

- Always be critical of the task you are told to do. Never assume the user is always right. This is a large project with many constraints
- Always focus on comparative analysis (multiple models)
- Always use MLflow for all experiments
- Always document every step for potential paper
- Always include "research prototype" disclaimers
- Always clean up temporary files after use
- Never claim clinical accuracy
- Never put .md files outside docs/ (except README and CLAUDE)
- Never commit PDFs or image files
- Never use emojis in code, documentation, or communication (unprofessional)
- Never create copies of existing files with modifications. Always modify original files to achieve the desired task.
