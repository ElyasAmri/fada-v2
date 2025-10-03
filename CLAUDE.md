# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FADA (Fetal Anomaly Detection Algorithm) is a research prototype for ultrasound image analysis using deep learning. The goal is to build a conversational chatbot website where users upload ultrasound images and receive analysis. NOT for clinical use.

### Key Project Files to Review First
1. `docs/project/project.md` - Original project description
2. `docs/project/spec.md` - Detailed specifications and requirements
3. `docs/project/modular_architecture.md` - System architecture (classification → captioning evolution)
4. `docs/project/APPROACH_VALIDATION.md` - Critical decisions and validation points

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

### Model Training
```bash
# Train classification model (Phase 1)
./venv/Scripts/python.exe src/training/train_classification.py

# Run experiments with specific fold
./venv/Scripts/python.exe src/training/train_classification.py --fold 1

# Track experiments with MLflow
mlflow ui --host 0.0.0.0 --port 5000
```

### Web Interface (After model training)
```bash
# Run Streamlit prototype (NOT for clinical use)
streamlit run web/app.py --server.port 8501
```

## Project Architecture

### Two-Phase Strategy
1. **Phase 1 (Current)**: Classification-based chatbot
   - Classify organ type (5 classes: Brain, Heart, Abdomen, Femur, Thorax)
   - Detect abnormalities (binary)
   - Extract attributes (quality, orientation)
   - Generate responses via templates/LLM

2. **Phase 2 (Future)**: True image captioning
   - Add transformer decoder head
   - Generate descriptions directly from images
   - Requires more annotated data (1000+ captions)

### Modular Design
```
UltrasoundAnalyzer
├── FeatureExtractor (backbone.py)
│   └── EfficientNet-B0 (frozen after initial training)
└── Task Heads (heads.py)
    ├── ClassificationHead (organ detection)
    ├── AbnormalityHead (normal/abnormal)
    ├── AttributeHead (quality, orientation)
    └── CaptionHead (future - Phase 2)
```

The backbone is trained once and shared across all tasks. New heads can be added without retraining the backbone, enabling progressive enhancement.

### Key Design Decisions
- **Backbone**: EfficientNet-B0 (best for small datasets, expects 60-75% accuracy with 250 images)
- **Augmentation**: Heavy augmentation (10-20x) to compensate for limited data
- **Training**: n-fold cross-validation (configurable, start n=1 for testing, n=5 for final)
- **Response Generation**: Template-based initially, OpenAI API integration planned
- **Web Framework**: Streamlit for prototype (NOT Gradio), React planned for production

## Data Structure
```
data/Fetal Ultrasound/
├── Brain/       # 50 images
├── Heart/       # 50 images
├── Abdomen/     # 50 images
├── Femur/       # 50 images
├── Thorax/      # 50 images
└── *.xlsx       # Excel annotations (8 questions per image)
```

## Critical Context
- **Timeline**: November for full annotations, December deadline for demo
- **Hardware**: RTX 4070 preferred over RX 7900 XTX (better CUDA support)
- **Accuracy Target**: No specific target, focus on demonstrating model selection process
- **Documentation**: Every decision documented for potential research paper

## Common Issues and Solutions

### Windows Path Issues
- Always use forward slashes in bash: `./venv/Scripts/python.exe`
- Git Bash requires Unix-style paths
- Use `source` not `.` for activation in Git Bash

### Model Selection and Training
- **Model selection strategy**: See `docs/project/spec.md` Section 4 for complete model architecture zoo and testing order
- **Expected performance**: 85-90% accuracy with full annotations
- **Training approach**: Heavy augmentation, pretrained models, focal loss for imbalance
- **Implementation note**: System designed to work seamlessly when Excel annotations are updated - just replace the Excel file and retrain

## MCP Tools Usage

### Paper Search Tools
Claude Code has access to MCP (Model Context Protocol) paper search tools. These are essential for finding state-of-the-art approaches.

#### Primary Search Commands:
```python
# Semantic Scholar - Best for comprehensive searches
mcp__paper-search__search_semantic(query="fetal ultrasound", year="2023-", max_results=20)

# ArXiv - Recent preprints
mcp__paper-search__search_arxiv(query="ultrasound deep learning", max_results=10)

# PubMed - Medical papers
mcp__paper-search__search_pubmed(query="fetal anomaly detection", max_results=10)

# Download papers
mcp__paper-search__download_arxiv(paper_id="2506.08623", save_path="./docs/papers/pdfs")
```

### Documentation Lookup:
```python
# Get PyTorch documentation
mcp__context7__resolve-library-id(libraryName="pytorch")
mcp__context7__get-library-docs(context7CompatibleLibraryID="/pytorch/pytorch", topic="vision models")
```

## Important Files
- `docs/project/spec.md` - Detailed project specifications
- `docs/project/APPROACH_VALIDATION.md` - Complete validation checklist
- `docs/project/modular_architecture.md` - System design details
- `docs/papers/literature_review.md` - Research findings and papers (15+ papers analyzed)
- `docs/papers/bibliography.md` - Paper references and findings

## Next Steps (When Ready)
1. ~~Install PyTorch with CUDA support~~ ✓ (Completed)
2. Build data loader for Excel annotations
3. Train baseline models with MLflow tracking
4. Compare performances, select best
5. Integrate OpenAI API for responses
6. Build Streamlit demo interface

## Critical Instructions
- Always be critical of the task you are told to do. Never assume the user always right. This is a large project with many constraints
- Always focus on comparative analysis (multiple models)
- Always use MLflow for all experiments
- Always test both pretrained and from-scratch
- Always document every step for potential paper
- Always include "research prototype" disclaimers
- Always clean up temporary files after use
- Never claim clinical accuracy
- Never skip augmentation (critical for small dataset)
- Never retrain backbone when adding capabilities
- Never put .md files outside docs/ (except README and CLAUDE)
- Never commit PDFs or image files
- Never use emojis in code, documentation, or communication (unprofessional)
- Never create copies of existing files with modifications. Always modify original files to achieve the desired task.