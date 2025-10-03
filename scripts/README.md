# Scripts

Utility and helper scripts for the FADA project.

## Structure

- **setup/** - Setup and configuration scripts
  - `setup_medgemma_access.py` - Configure MedGemma model access
  - `check_medgemma_status.py` - Verify MedGemma setup status

- **utilities/** - Helper utilities
  - `clear_cuda.py` - Clear CUDA cache
  - `count_labeled_images.py` - Count labeled dataset images
  - `evaluate_all_vqa.py` - Evaluate VQA models

- **training/** - Training orchestration
  - `train_all_full_scale.py` - Full-scale training script

## Usage

Run scripts from project root:
```bash
./venv/Scripts/python.exe scripts/utilities/clear_cuda.py
./venv/Scripts/python.exe scripts/training/train_all_full_scale.py
```
