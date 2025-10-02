# BLIP-2 VQA Training Notebooks

This directory contains all BLIP-2 Visual Question Answering training notebooks for the FADA project.

## Contents

### Training Notebooks
- `train_blip2_1epoch.ipynb` - Original Non_standard_NT training (487 images)
- `train_blip2_abdomen.ipynb` - Abdomen training (2424 images)
- `train_blip2_femur.ipynb` - Femur training (1165 images)
- `train_blip2_thorax.ipynb` - Thorax training (1793 images)
- `train_blip2_standard_nt.ipynb` - Standard_NT training (1508 images)
- `train_blip2_cervix.ipynb` - Cervix (awaiting labeled data)
- `train_blip2_trans_cerebellum.ipynb` - Trans-cerebellum (awaiting labeled data)
- `train_blip2_trans_thalamic.ipynb` - Trans-thalamic (awaiting labeled data)
- `train_blip2_trans_ventricular.ipynb` - Trans-ventricular (awaiting labeled data)
- `train_blip2_non_standard_nt_full.ipynb` - Full-scale training attempt

**Note**: Notebooks retain execution history from training runs. Use "Clear All Outputs" before running to start fresh.

## Quick Start

### Run Single Category (Validation)
```bash
# 5 images, 1 epoch for quick testing
jupyter notebook train_blip2_abdomen.ipynb
```

### Run with Papermill (Automated)
```bash
# Full dataset, 3 epochs
python -m papermill \
    train_blip2_abdomen.ipynb \
    train_blip2_abdomen_full.ipynb \
    -p num_images 2424 \
    -p num_epochs 3
```

### Run All Categories
```bash
# From project root
python train_all_full_scale.py --epochs 3 --sort-by-size
```

## Training Parameters

Default parameters (can be overridden with papermill):
- `num_images`: Number of images to train on (default: 5 for validation)
- `num_epochs`: Training epochs (default: 1)
- `batch_size`: Batch size (default: 1)
- `learning_rate`: Learning rate (default: 1e-4)

## Model Architecture

**Base Model**: Salesforce/blip2-opt-2.7b
**Fine-tuning**: LoRA (Low-Rank Adaptation)
- r=8
- alpha=16
- target_modules: ["q_proj", "v_proj"]
- dropout=0.05

**Quantization**: 8-bit (for memory efficiency)

## Training Results

### Validation Models (5 images, 1 epoch)
| Category | Status | Training Time | Model Path |
|----------|--------|---------------|------------|
| Non_standard_NT | ✅ | ~1 min | `outputs/blip2_1epoch/final_model` |
| Abdomen | ✅ | ~1 min | `outputs/blip2_abdomen/final_model` |
| Femur | ✅ | ~1 min | `outputs/blip2_femur/final_model` |
| Thorax | ✅ | ~1 min | `outputs/blip2_thorax/final_model` |
| Standard_NT | ✅ | ~1 min | `outputs/blip2_standard_nt/final_model` |

### Full-Scale Training (All images, 3 epochs)
Training times (estimated):
- Non_standard_NT (487 images): ~30-45 min
- Femur (1165 images): ~1-1.5 hours
- Standard_NT (1508 images): ~1.5-2 hours
- Thorax (1793 images): ~2-2.5 hours
- Abdomen (2424 images): ~3-3.5 hours

**Total**: ~8-10 hours for all categories

## Data Format

Each category requires:
1. **Images**: `data/Fetal Ultrasound/{Category}/`
2. **Annotations**: `data/Fetal Ultrasound Labeled/{Category}_image_list.xlsx`

Excel file must contain:
- Column: `Image Name`
- Columns: `Q1`, `Q2`, ..., `Q8` (8 standard medical questions with answers)

## Standard Questions

1. Anatomical Structures: List all visible anatomical structures
2. Fetal Orientation: Describe the orientation of the fetus
3. Plane Evaluation: Assess if image is at standard diagnostic plane
4. Biometric Measurements: Identify biometric measurements
5. Gestational Age: Estimate gestational age range
6. Image Quality: Evaluate image quality and clarity
7. Normality/Abnormality: Identify visible abnormalities
8. Clinical Recommendations: Provide clinical recommendations

## Output Structure

Each training run creates:
```
outputs/blip2_{category}/
├── final_model/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors  # LoRA adapters (~10MB)
│   ├── tokenizer.json
│   └── processor_config.json
├── checkpoint-*/                   # Training checkpoints
└── training_summary.json           # Training metrics
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` to 1
- Close other GPU applications
- Use `clear_cuda.py` before training

### Training Too Slow
- Check GPU is being used (not CPU)
- Verify CUDA is available
- Use smaller `num_images` for testing

### Model Not Found Error
- Check `data/Fetal Ultrasound Labeled/` for Excel file
- Verify Excel file name matches category
- Ensure images exist in `data/Fetal Ultrasound/{Category}/`

## Related Files

- `../../train_all_full_scale.py` - Automated training script
- `../../evaluate_all_vqa.py` - Evaluation script
- `../../test_vqa_category.py` - Single category testing
- `../../info/vqa_training_summary.md` - Detailed training documentation
- `../../info/vqa_usage_guide.md` - Complete usage guide

## Citation

If using these models for research:
```
FADA: Fetal Anomaly Detection Algorithm
Visual Question Answering with BLIP-2
https://github.com/[your-repo]
```
