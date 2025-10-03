# Experiments

This directory contains all experimental work for the FADA project, including VLM testing, notebooks, and external model repositories.

## Structure

- **vlm_testing/** - Vision-Language Model testing and evaluation
  - **comprehensive/** - Latest comprehensive VLM tests (13 models tested Oct 2025)
  - **quick_tests/** - Early quick testing scripts
  - **legacy/** - Historical VLM test scripts
  - **results/** - VLM test results (JSON, CSV files)

- **notebooks/** - Jupyter notebooks for exploratory work
  - **blip2_training/** - BLIP-2 model training experiments
  - **experiments/** - General experimental notebooks
  - **exploratory/** - Standalone exploratory notebooks

- **external_models/** - Cloned repositories for comparison/testing
  - **FetalCLIP/** - FetalCLIP model repository
  - **TinyGPT-V/** - TinyGPT-V model repository

## Usage

All experimental scripts should be run from the project root:
```bash
./venv/Scripts/python.exe experiments/vlm_testing/comprehensive/test_model.py
```

## Notes

- This directory is tracked in git
- Large model weights and outputs are stored in `artifacts/` (gitignored)
- See `docs/experiments/` for detailed documentation of experimental results
