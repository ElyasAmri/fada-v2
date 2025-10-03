# Fetal Ultrasound Analysis Chatbot - Web Interface

## Overview
This is a Streamlit-based web interface for the Fetal Ultrasound Analysis Chatbot. It provides an interactive platform for analyzing ultrasound images and receiving AI-generated explanations.

## Features
- **Single Image Analysis**: Upload and analyze individual ultrasound images
- **Batch Processing**: Analyze multiple images at once
- **Interactive Chat**: Ask follow-up questions about the analysis
- **OpenAI Integration**: Enhanced natural language responses using GPT
- **Confidence Visualization**: Clear indication of prediction confidence
- **Educational Information**: Detailed explanations of each anatomical view

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure OpenAI API (Optional but Recommended)
Create a `.env.local` file in the project root:
```bash
cp .env.local.template .env.local
```

Edit `.env.local` and add your OpenAI API key and model preference:
```
OPENAI_API_KEY=your_actual_api_key_here
MODEL_NAME=gpt-4o-mini  # Options: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4
TEMPERATURE=0.6          # Lower for medical consistency
MAX_TOKENS=800          # Longer for detailed explanations
```

#### Available GPT Models (2024)
- **gpt-4o** - Most capable multimodal model (highest accuracy)
- **gpt-4o-mini** - Balanced performance and cost (recommended)
- **gpt-4-turbo** - Previous generation, still very capable
- **gpt-4** - Original GPT-4, consistent but slower
- **gpt-3.5-turbo** - Budget option, fast but less accurate

### 3. Ensure Model is Trained
The web interface requires a trained model at:
```
models/best_model_efficientnet_b0_12class.pth
```

If not present, train the model using:
```bash
python src/scripts/train_12class.py
```

## Running the Application

### Start the Streamlit App
```bash
streamlit run web/app.py
```

The app will open in your browser at `http://localhost:8501`

### Alternative Ports
To use a different port:
```bash
streamlit run web/app.py --server.port 8080
```

## Usage Guide

### Main Chat Interface
1. Upload an ultrasound image (PNG/JPG)
2. Optionally add a specific question
3. Click "Analyze Image"
4. View classification results and AI response
5. Ask follow-up questions if needed

### Batch Analysis
1. Go to "Batch Analysis" tab
2. Upload multiple images
3. Click "Analyze All"
4. View summary statistics and individual results

### Settings (Sidebar)
- **Confidence Threshold**: Adjust minimum confidence for definitive statements
- **OpenAI Integration**: Toggle GPT-enhanced responses
- **Clear Conversation**: Reset the chat history

## Response Types

### Template-Based (Default)
- Works without OpenAI API
- Structured, consistent responses
- Clinical information for all 12 views

### OpenAI-Enhanced (When API Key Configured)
- More natural, conversational responses
- Better handling of follow-up questions
- Context-aware explanations

## Supported Anatomical Views (12 Classes)
1. **Brain Views**: Trans-thalamic, Trans-ventricular, Trans-cerebellum
2. **Cardiac**: Aorta, Thorax
3. **Growth**: Abdomen, Femur
4. **Maternal**: Cervix, Cervical
5. **Screening**: Standard_NT, Non_standard_NT
6. **Labor**: Public_Symphysis_fetal_head

## Confidence Levels
- **High (>70%)**: Definitive analysis with detailed information
- **Medium (50-70%)**: Qualified statements with some uncertainty
- **Low (<50%)**: Uncertain, professional verification recommended

## Important Notes

### Medical Disclaimer
This is a **research prototype** for educational purposes only:
- NOT approved for clinical use
- NOT FDA-approved
- Always consult healthcare professionals
- Do not use for medical diagnosis

### Performance
- Model accuracy: ~90% on test set
- Processing time: 1-3 seconds per image
- GPU recommended for faster inference

### Privacy
- Images are processed locally
- OpenAI API calls (if enabled) send only text descriptions
- No image data is stored permanently

## Troubleshooting

### Model Not Found
```
Error: Model not found at models/best_model_efficientnet_b0_12class.pth
```
Solution: Train the model first using the training script

### OpenAI API Errors
```
OpenAI API not available
```
Solution: Check your API key in `.env.local`

### CUDA/GPU Issues
The app will automatically fall back to CPU if GPU is not available.

### Memory Issues
For batch processing of many images:
- Process in smaller batches
- Reduce image size if needed
- Close other applications

## Development

### Project Structure
```
web/
├── app.py              # Main Streamlit application
├── README.md           # This file
└── assets/            # Static assets (if needed)

src/chatbot/
├── chatbot.py         # Core chatbot logic
├── response_generator.py  # Template responses
└── openai_integration.py  # GPT integration
```

### Adding New Features
1. Modify `app.py` for UI changes
2. Update `chatbot.py` for analysis logic
3. Edit response templates in `response_generator.py`

## Future Enhancements
- [ ] Add abnormality detection
- [ ] Implement measurement tools
- [ ] Add report generation
- [ ] Support DICOM format
- [ ] Multi-language support
- [ ] Export analysis results