# VLM Test Scripts

This directory contains all test scripts for vision-language models (VLMs) tested for the FADA project's fetal ultrasound VQA task.

## Overview

**Total Scripts**: 22
**Testing Period**: October 1-3, 2025
**Purpose**: Evaluate VLM models for fetal ultrasound image analysis
**Hardware**: RTX 4070 8GB VRAM, Windows 11

## Script Categories

### Working Models (Successfully Tested)
- `test_kosmos2.py` - Microsoft Kosmos-2 (100% fetal context, 33% anatomy)
- `test_llava_next_4bit.py` - LLaVA-NeXT-7B with 4-bit quantization (~50% overall)
- Florence-2 variants (multiple attempts):
  - `test_florence2_simple.py`
  - `test_florence2_quick.py`
  - `test_florence2_fixes.py`
  - `test_florence2_bypass.py`
  - `test_florence2_new_env.py`

### Failed Models (Compatibility/Architecture Issues)
- `test_minicpm_v.py` - MiniCPM-V (API mismatch)
- `test_cogvlm.py` - CogVLM (requires triton - Linux only)
- `test_deepseek_vl.py` - DeepSeek-VL (custom architecture)
- `test_fuyu.py` - Fuyu-8B (memory exceeded)
- `test_mplug_owl2.py` - mPLUG-Owl2 (custom architecture)
- `test_tinygpt_v.py` - TinyGPT-V (missing config)
- `test_idefics2.py` - IDEFICS2-8B (37.5% - below BLIP-2)
- `test_chexagent.py` - CheXagent-8b (0% - chest X-ray only)
- `test_qwen25_vl.py` - Qwen2.5-VL-3B (inference error)

### Qwen-VL Variants (Visual Encoder Issues)
- `test_qwen_vl.py` - Original Qwen-VL
- `test_qwen_vl_fixed.py` - Attempted fix
- `test_qwen_vl_extended.py` - Extended testing
- `test_qwen_vl_int4.py` - 4-bit quantization attempt

### Other
- `test_medical_models.py` - Various medical VLMs
- `test_minigpt4.py` - MiniGPT-4 (requires GitHub clone)

## Usage

All scripts follow the same pattern:
```bash
# From project root
./venv/Scripts/python.exe scripts/vlm_tests/test_<model>.py
```

## Data Directory

Scripts expect fetal ultrasound data at:
```
data/Fetal Ultrasound/
├── Abodomen/
├── Aorta/
├── Brain/
├── Femur/
├── Heart/
└── ... (other categories)
```

## Key Results

**Best Model**: BLIP-2 (~55% accuracy) - tested separately
**Runner-ups**:
- LLaVA-NeXT-7B: ~50%
- Kosmos-2: 44% (100% fetal context recognition)

**Complete results**: See `info/complete_vlm_testing_results.md`

## Notes

- Most scripts use 4-bit quantization for large models (>6B params)
- Scripts are preserved for documentation and reproducibility
- Some scripts have multiple variants due to troubleshooting attempts
- All failed models documented with error details
