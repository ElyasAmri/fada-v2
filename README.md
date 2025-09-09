# FADA - Fetal Anomaly Detection Algorithm

A research prototype for ultrasound image analysis using deep learning. This project builds a conversational interface where users can upload fetal ultrasound images and receive AI-powered analysis.

**⚠️ IMPORTANT: This is a research prototype for educational purposes only. NOT for clinical use.**

## Project Structure

```
fada-v3/
├── src/           # Source code (models, training, data loaders)
├── papers/        # Literature review and bibliography  
├── data/          # Ultrasound images and annotations (not tracked)
└── info/          # Project documentation (internal)
```

## Setup

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (RTX 4070 or similar)
- Windows/Linux/macOS

### Installation

1. Create virtual environment:
```bash
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash
# or
venv\Scripts\activate  # Windows CMD
```

2. Install PyTorch with CUDA support:
```bash
pip install -r requirements.txt
```

## Dataset

The project uses 250 annotated fetal ultrasound images across 5 organ types:
- Brain (50 images)
- Heart (50 images)  
- Abdomen (50 images)
- Femur (50 images)
- Thorax (50 images)

Each image includes structured annotations with 8 medical assessment questions.

## Approach

### Phase 1: Classification-Based Analysis (Current)
- Multi-class organ classification
- Abnormality detection
- Template/LLM-based response generation

### Phase 2: Image Captioning (Future)
- Direct description generation from images
- Requires additional annotated data

## Key Technologies

- **Deep Learning Framework**: PyTorch
- **Primary Model**: EfficientNet-B0
- **Experiment Tracking**: MLflow
- **Web Interface**: Streamlit (planned)

## Expected Performance

With 250 images (50 per class):
- Target accuracy: 60-75%
- Focus on comparative model analysis
- Heavy data augmentation to compensate for limited dataset

## Research Papers

See `papers/` directory for:
- Literature review of 15+ relevant papers
- Bibliography of key references
- State-of-the-art approaches analysis

## Disclaimer

This is a research and educational project. All outputs should include:
"For educational purposes only. Not for clinical use. Consult healthcare provider for medical advice."

## License

Research prototype - not for production use.