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

- **training/** - Training scripts
  - `train_12class.py` - 12-class classifier training with MLflow
  - `train_all_full_scale.py` - Full-scale training orchestration

- **analysis/** - Data analysis tools
  - `analyze_dataset.py` - Dataset structure and content analysis

## Usage

Run scripts from project root:
```bash
./venv/Scripts/python.exe scripts/utilities/clear_cuda.py
./venv/Scripts/python.exe scripts/training/train_all_full_scale.py
```
