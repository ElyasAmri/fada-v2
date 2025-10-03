# FADA Web Application

Multi-Model VLM Comparison Interface for Fetal Ultrasound Analysis

## Overview

This web application demonstrates the FADA multi-model VQA workflow:
1. User (sonographer) uploads ultrasound image
2. System detects organ type automatically
3. Top-3 VLM models answer 8 standard questions
4. Sonographer selects best answer for each question
5. Selections saved for analysis and model improvement

## Applications

### `app_mvp.py` - Multi-Model Comparison (MVP) ⭐
**Use this for demo and funding proposal**

Features:
- Upload ultrasound images
- Automatic organ type detection
- Run 3 top VLM models in parallel: MiniCPM-V-2.6 (88.9%), Qwen2-VL-2B (83.3%), InternVL2-4B (82%)
- Side-by-side answer comparison
- User selection interface (radio buttons per question)
- Save selections as JSON for analysis
- **Modular design**: Can swap local GPU inference with API endpoints

Run:
```bash
streamlit run web/app_mvp.py --server.port 8501
```

### `app.py` - Legacy Single-Model Interface
**Old version using BLIP-2** - kept for reference

Run:
```bash
streamlit run web/app.py --server.port 8501
```

## Architecture

### Modular Design for API Migration

The MVP uses a modular architecture that allows easy migration from local GPU to cloud API:

```
VLMInterface (Abstract)
├── LocalVLM (GPU inference)
│   └── HuggingFace models with 4-bit quantization
└── APIVLM (Cloud inference) [Future]
    └── POST requests to inference endpoint
```

**Current**: Local GPU inference on RTX 4070 (8GB)
**Future**: Swap to cloud API with dedicated GPU (easy one-line change)

### Question Management

Questions are loaded dynamically from Excel annotation files:
- Located in `data/Fetal Ultrasound/*_image_list.xlsx`
- Automatically extracted from column headers
- Update Excel → Questions update automatically
- No code changes needed

## Key Features

### 1. Expert-in-the-Loop
- Sonographer validates all AI outputs
- No single model failure point
- Real-world preference data collection

### 2. Model Comparison
- Top-3 models run on same image
- Side-by-side answer display
- Selection tracking per question

### 3. Data Collection
- User selections saved as JSON
- Track model performance by question type
- Build ensemble/meta-model from preferences

### 4. Future-Proof Design
- Easy swap to API endpoints
- Configurable model list
- Extensible to more models (Top-5, Top-10)

## Configuration

### Local GPU Mode (Default)
```python
# In sidebar
☐ Use API Endpoints
```

Models run sequentially:
- Load model → Answer questions → Unload → Next model
- Total time: 30-60 seconds for 3 models
- Memory: ~5GB max per model (4-bit quantization)

### Cloud API Mode (Future)
```python
# In sidebar
☑ Use API Endpoints
API Endpoint: http://your-api.com
API Key: ***********
```

Models run via API:
- Parallel requests possible
- No local GPU memory needed
- Faster response (~5-10 seconds total)

## Output Data

User selections saved to: `outputs/user_selections/selection_YYYYMMDD_HHMMSS.json`

Format:
```json
{
  "timestamp": "20251003_143025",
  "image_name": "Brain_042.png",
  "detected_type": "Brain",
  "selections": {
    "0": "minicpm",      // Q1: Anatomical Structures
    "1": "qwen2vl",      // Q2: Fetal Orientation
    "2": "minicpm",      // Q3: Plane Evaluation
    "3": "internvl2",    // Q4: Biometric Measurements
    "4": "minicpm",      // Q5: Gestational Age
    "5": "qwen2vl",      // Q6: Image Quality
    "6": "minicpm",      // Q7: Normality/Abnormality
    "7": "minicpm"       // Q8: Clinical Recommendations
  }
}
```

Analysis:
- Track which model performs best per question type
- Identify model strengths/weaknesses
- Build model ensemble based on preferences
- Train meta-model to predict best answer

## Dependencies

Installed via `requirements.txt`:
- streamlit
- transformers
- torch
- bitsandbytes (4-bit quantization)
- pandas
- openpyxl (Excel reading)
- pillow

## Performance

### Local GPU (RTX 4070 8GB)
- **Sequential Loading**: ~30-60 seconds for 3 models
- **Memory**: ~5GB per model (4-bit quantization)
- **Quality**: No degradation from quantization

### Cloud API (Future)
- **Parallel Requests**: ~5-10 seconds for 3 models
- **Memory**: None (server-side)
- **Cost**: Pay-per-request pricing

## Migration to Cloud API

To switch from local to cloud API:

1. Deploy inference endpoint (e.g., AWS SageMaker, Azure ML, GCP Vertex AI)
2. In app, enable "Use API Endpoints" in sidebar
3. Enter API endpoint URL and key
4. Done! ✅

No code changes needed - the `VLMInterface` abstraction handles it.

## Next Steps

### For Funding Proposal
- [x] MVP web app (this app)
- [ ] Run demo with sample images
- [ ] Collect screenshots/video
- [ ] Document workflow
- [ ] Submit funding request for:
  - Dedicated training machine (large GPU)
  - Cloud inference endpoint

### After Funding
- [ ] Deploy cloud API endpoint
- [ ] Fine-tune models on FADA dataset (target: 95%+)
- [ ] Add more models (Top-5)
- [ ] Implement ensemble based on user selections
- [ ] Add confidence scores
- [ ] Production deployment

## Notes

- **Research Prototype**: Not for clinical use
- **Expert Validation Required**: All AI outputs reviewed by sonographer
- **Data Privacy**: Images processed locally (unless using API mode)
- **Model Updates**: Questions update automatically from Excel files

---

*For questions or issues, see project documentation in `docs/`*
