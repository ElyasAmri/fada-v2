# Experiments

This directory contains all experimental work for the FADA project, including VLM testing, notebooks, and external model integrations.

## Active Directories

### api_models/
API-based VLM inference infrastructure.
- `test_api_vlm.py` - Main async parallel VLM testing (Gemini, OpenAI, Grok)
- `batch_gemini_annotation.py` - Batch API processing for Gemini
- `results/` - Checkpoints, batch files, and results (DO NOT DELETE)

### vlm_testing/
Local Vision-Language Model testing.
- `comprehensive/` - SOTA model tests (InternVL2, Qwen2-VL, MolMo, etc.)
- `quick_tests/` - Fast validation scripts
- `medgemma/` - Medical VLM-specific tests

### external_models/
Third-party model integrations.
- `FetalCLIP/` - MBZUAI FetalCLIP (zero-shot, probing, few-shot)

### fine_tuning/
Model fine-tuning scripts.
- `train_qwen3vl_lora.py` - LoRA fine-tuning for Qwen3-VL
- `train_vastai.py` - Training on Vast.ai infrastructure

### notebooks/
Jupyter notebooks for training.
- `blip2_training/` - BLIP-2 VQA training (5 categories complete)

## Usage

Run scripts from project root:
```bash
./venv/Scripts/python.exe experiments/api_models/test_api_vlm.py --help
./venv/Scripts/python.exe experiments/vlm_testing/comprehensive/test_internvl2.py
```

## Notes

- Results in `api_models/results/` are critical - do not delete
- See `docs/experiments/` for detailed documentation
- Large model weights stored in `artifacts/` (gitignored)
