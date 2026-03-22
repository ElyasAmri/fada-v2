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

# Fine-tune with LoRA (cloud, RCCG A100/H100)
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
- **Cloud training**: RCCG (A100 at $1.35/h, H100 at $1.90/h)
- **Web Framework**: Streamlit for prototype

## Data Structure

```
data/Fetal Ultrasound/
+-- Abdomen/             # 2,424 images
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
data/Fetal Ultrasound Annotations Normalized.xlsx  # Master GT annotations (18,936 rows)
data/vlm_training/       # Sonographer GT files (gt_train/val/test.jsonl, ShareGPT variants)
data/dataset_splits.json # Train/val/test split (15,231 / 1,894 / 1,894)
data/archive/            # Gemini pseudo-labels, legacy training formats, old annotations
```

## Critical Context

- **Hardware**: RTX 5090 (24GB VRAM) local, cloud GPUs for benchmarking
- **Accuracy**: Best verified GT score is 81.1% (Qwen2.5-VL-7B fine-tuned)
- **Key caveat**: All current scores are GT-based (embedding similarity + per-question metrics). Phase 1 proxy scores were archived.
- **Documentation**: Every decision documented for research paper

## Common Issues and Solutions

### Windows Path Issues

- Always use forward slashes in bash: `./venv/Scripts/python.exe`
- Git Bash requires Unix-style paths
- Use `source` not `.` for activation in Git Bash

### Cross-Platform Image Paths

- JSONL training files use relative paths (e.g., `Abdomen/Abdomen_001.png`)
- Training script `--data-root` prepends the correct base directory at runtime

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

## RCCG Cloud Cluster

H100 machines managed via Ansible through WSL. Instance count varies.

### CRITICAL: Always use ./r.sh -- NEVER raw SSH

**NEVER use raw `ssh -i ~/.ssh/rccg_key ubuntu@<ip>` commands.** Always use `./r.sh` from project root:

```bash
# Simple commands (no quoting needed)
./r.sh ssh fada-1 hostname
./r.sh ssh fada-1 nvidia-smi

# Complex commands with pipes/redirects (use heredoc, no quoting needed)
./r.sh ssh fada-1 <<'CMD'
nvidia-smi | grep MiB && echo done
CMD

# Status / logs
./r.sh status
./r.sh logs fada-1
./r.sh vllm-log fada-1
./r.sh queue-log fada-1

# Pull checkpoints
./r.sh pull fada-1

# Run Ansible playbooks
./r.sh play setup --limit fada-1
./r.sh play run_eval --limit fada-1
./r.sh play run_queue --limit fada-1,fada-2

# Launch long-running jobs (SCP-based, survives SSH drops)
experiments/rccg/launch_job.sh fada-1 "training command"

# Quick cluster status
experiments/rccg/check_status.sh
```

A PreToolUse hook (`.claude/hooks/block-raw-ssh.sh`) enforces this -- raw SSH commands are blocked.

### Key Files

- `r.sh` - Project root shortcut to `experiments/rccg/rccg.sh`
- `experiments/rccg/rccg.sh` - Unified CLI wrapper (resolves host -> IP from inventory, supports stdin for complex commands)
- `experiments/rccg/inventory/hosts.yml` - Machine IPs, models, volume devices
- `experiments/rccg/check_status.sh` - Parallel SSH status check across all hosts
- `experiments/rccg/launch_job.sh` - SCP-based job launcher (tmux, logs, jobs.json tracking)
- `experiments/rccg/jobs.json` - Active job metadata
- `experiments/rccg/playbooks/` - Ansible playbooks (setup, run_eval, run_queue, stop, exec, status, collect, deploy_monitor, deploy_tunnel)
- `experiments/rccg/fada-monitor/` - Process monitor server (see below)

### Model Rotation Workflow

1. Edit `vllm_model` and `vllm_models` list in `inventory/hosts.yml`
2. `./r.sh play run_queue --limit <host>` (queue system: stops old vLLM, downloads model, starts vLLM, runs eval for each model in list)
3. `./r.sh status` to monitor progress
4. `./r.sh pull <host>` to fetch checkpoints when done

### Process Monitor (fada-monitor)

Lightweight HTTP server running on RCCG machines that watches ML processes, tracks GPU stats, and queues completion events. Accessible via Cloudflare Tunnel -- no open ports or SSH needed.

**URL**: `https://fada-monitor.elyasamri.com`

**Endpoints**:

- `GET /health` - Server liveness, watched process count, pending event count
- `GET /processes` - Monitored processes with CPU/mem/runtime
- `GET /events` - Unprocessed completion events (process_started, process_completed)
- `POST /events/ack` - Acknowledge events (body: `{"ids": ["..."]}`)
- `GET /gpu` - nvidia-smi summary (utilization, memory, temperature)

**Architecture**:

- `experiments/rccg/fada-monitor/monitor.py` - Single-file Flask server, `/proc` scanning, event queue
- `experiments/rccg/fada-monitor/monitor_config.json` - Watch patterns (eval_hf_peft.py, launch_eval.sh, etc.)
- Runs as systemd service (`fada-monitor.service`) on the remote
- Cloudflare Tunnel (`cloudflared`) exposes localhost:9731 to the public URL

**Claude Code Integration** (`.claude/hooks/`):

- `check-remote-events.sh` - Curls `/events`, acks consumed events, writes `event_result.json`
- Registered as both `Stop` and `SessionStart` hook in `.claude/settings.local.json`
- Stop hook: if eval finishes while Claude is on, triggers continuation (exit 2)
- SessionStart hook: if eval finished while Claude was off, triggers response on startup
- `wait-for-remote.sh` - SSH-based polling hook (marker file pattern, for manual use)

**Deployment**:

```bash
# Deploy/update monitor server
experiments/rccg/r.sh play deploy_monitor --limit fada-1

# Deploy/update Cloudflare Tunnel (token in experiments/rccg/fada-monitor/.tunnel_token)
TOKEN=$(cat experiments/rccg/fada-monitor/.tunnel_token) && experiments/rccg/r.sh play deploy_tunnel --limit fada-1 -e "tunnel_token=$TOKEN"
```

### Scoring Pipeline

After pulling checkpoints:

```bash
./venv/Scripts/python.exe experiments/evaluation/checkpoint_to_predictions.py --checkpoint <checkpoint.json> --output <predictions.jsonl>
./venv/Scripts/python.exe experiments/evaluation/score_against_gt.py --predictions <predictions.jsonl> --output <scores.json>
```

## Critical Instructions

- **NEVER fine-tune a model that has already been fine-tuned** unless there is a specific, reasonable justification (e.g., fixing a known training issue, testing a hypothesis that could meaningfully improve scores). Hyperparameter variants are NOT acceptable as a lazy fallback when a different model fails. The priority is always to fine-tune NEW, UNIQUE base models. When a non-Qwen model fails, replace it with another non-Qwen model, not a Qwen variant.
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
