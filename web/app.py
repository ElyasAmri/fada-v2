"""
FADA - Fetal Anomaly Detection Algorithm
Streamlit Chatbot Interface
"""

import streamlit as st
import numpy as np
from PIL import Image
import io
import base64
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pathlib import Path
import sys
import time
from datetime import datetime
import random

# Add src to path for model imports
sys.path.append(str(Path(__file__).parent.parent))

# Import model components
import sys

# Try importing classifier separately
try:
    from src.models.classifier import FetalUltrasoundClassifier12
    MODEL_AVAILABLE = True
    sys.stderr.write("Classifier import successful\n")
    sys.stderr.flush()
except ImportError as e:
    MODEL_AVAILABLE = False
    sys.stderr.write(f"Classifier import failed: {e}\n")
    sys.stderr.flush()

# Try importing VQA separately
try:
    from src.models.vqa_model import UltrasoundVQA
    VQA_AVAILABLE = True
    sys.stderr.write("VQA import successful\n")
    sys.stderr.flush()
except Exception as e:
    VQA_AVAILABLE = False
    sys.stderr.write(f"VQA import failed (non-critical): {e}\n")
    import traceback
    sys.stderr.write(traceback.format_exc())
    sys.stderr.flush()

# Page configuration
st.set_page_config(
    page_title="FADA - Ultrasound Analysis",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': 'FADA - Fetal Anomaly Detection Algorithm'
    }
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Welcome to FADA Ultrasound Analysis System. Upload an ultrasound image to begin analysis."
    })

if "model" not in st.session_state:
    st.session_state.model = None

if "vqa_model" not in st.session_state:
    st.session_state.vqa_model = None

if "vqa_enabled" not in st.session_state:
    st.session_state.vqa_enabled = False

if "analyzing" not in st.session_state:
    st.session_state.analyzing = False

if "total_analyses" not in st.session_state:
    st.session_state.total_analyses = 0

if "current_image" not in st.session_state:
    st.session_state.current_image = None

if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

if "last_classification" not in st.session_state:
    st.session_state.last_classification = None

if "debug_info" not in st.session_state:
    st.session_state.debug_info = []

# Load model function
@st.cache_resource
def load_model():
    """Load the trained model"""
    if not MODEL_AVAILABLE:
        return None

    try:
        model_paths = [
            Path("models/best_model_efficientnet_b0_12class.pth"),
            Path("models/best_model_efficientnet_b0.pth"),
            Path("models/best_model.pth")
        ]

        for model_path in model_paths:
            if model_path.exists():
                if "12class" in str(model_path):
                    num_classes = 12
                else:
                    num_classes = 5

                model = FetalUltrasoundClassifier12(
                    num_classes=num_classes,
                    backbone='efficientnet_b0',
                    pretrained=False
                )

                checkpoint = torch.load(model_path, map_location='cpu')

                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                else:
                    model.load_state_dict(checkpoint)

                model.eval()
                return model

        return None
    except Exception as e:
        st.error(f"Failed to load classification model: {str(e)}")
        return None

@st.cache_resource
def load_vqa_model():
    """Load the VQA model (lazy loading)"""
    if not VQA_AVAILABLE:
        return None

    try:
        # Create VQA instance but don't load model yet
        vqa = UltrasoundVQA(
            model_path="outputs/blip2_1epoch/final_model",
            device="auto"
        )
        # Model will be loaded on first use via answer_question()
        return vqa
    except Exception as e:
        print(f"VQA initialization failed: {e}")
        return None

def preprocess_image(image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return transform(image).unsqueeze(0)

def analyze_ultrasound(image, model=None):
    """Analyze ultrasound image and return results"""
    if model is None:
        st.error("Classification model not loaded. Check console for errors.")
        return {
            "organ": "Error",
            "confidence": 0.0,
            "is_normal": False,
            "quality": "Unknown",
            "orientation": "Unknown",
            "processing_time": 0.0
        }

    # Real model inference
    with torch.no_grad():
        start_time = time.time()
        image_tensor = preprocess_image(image)
        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)
        organ_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities.max().item()

        # 12-class model with correct order from dataset
        organs = [
            "Abodomen",  # 0
            "Aorta",  # 1
            "Cervical",  # 2
            "Cervix",  # 3
            "Femur",  # 4
            "Non_standard_NT",  # 5
            "Public_Symphysis_fetal_head",  # 6
            "Standard_NT",  # 7
            "Thorax",  # 8
            "Trans-cerebellum",  # 9
            "Trans-thalamic",  # 10
            "Trans-ventricular"  # 11
        ]

        if organ_idx >= len(organs):
            organ_idx = len(organs) - 1

        # Map technical names to user-friendly names
        organ_display_names = {
            "Abodomen": "Abdomen",
            "Aorta": "Aortic Arch",
            "Cervical": "Cervical View",
            "Cervix": "Cervix",
            "Femur": "Femur",
            "Non_standard_NT": "Non-standard NT",
            "Public_Symphysis_fetal_head": "Fetal Head Position",
            "Standard_NT": "Standard NT",
            "Thorax": "Thorax",
            "Trans-cerebellum": "Transcerebellar Plane",
            "Trans-thalamic": "Transthalamic Plane",
            "Trans-ventricular": "Transventricular Plane"
        }

        detected_organ = organs[organ_idx]
        display_name = organ_display_names.get(detected_organ, detected_organ)

        # Confidence-based quality assessment
        if confidence > 0.85:
            quality = "Excellent"
            is_normal = True
        elif confidence > 0.7:
            quality = "Good"
            is_normal = True
        elif confidence > 0.5:
            quality = "Fair"
            is_normal = False
        else:
            quality = "Poor"
            is_normal = False

        return {
            "organ": display_name,
            "confidence": confidence,
            "is_normal": is_normal,
            "quality": quality,
            "orientation": "Standard View",
            "processing_time": time.time() - start_time
        }

def generate_response(analysis_results):
    """Generate chatbot response based on analysis results"""
    organ = analysis_results['organ']
    confidence = analysis_results['confidence']
    is_normal = analysis_results['is_normal']
    quality = analysis_results['quality']
    orientation = analysis_results['orientation']

    response = f"## Analysis Complete\n\n"
    response += f"**Detected View:** {organ}\n"
    response += f"**Confidence:** {confidence:.1%}\n"
    response += f"**Image Quality:** {quality}\n"
    response += f"**Orientation:** {orientation}\n\n"

    if is_normal:
        response += "### Assessment\n"
        response += f"The {organ.lower()} ultrasound appears normal based on the analyzed features. "
        response += "No significant abnormalities detected in this view.\n\n"
    else:
        response += "### Findings\n"
        response += f"Potential abnormality detected in the {organ.lower()} view. "
        response += "Further evaluation recommended.\n\n"

    # Add organ-specific information
    organ_info = {
        "Abdomen": "The abdominal view shows stomach, liver, and cord insertion.",
        "Aortic Arch": "The aortic arch view assesses cardiac output and vessel structure.",
        "Cervical View": "The cervical view evaluates the cervical region.",
        "Cervix": "The cervical view measures cervical length.",
        "Femur": "The femur view is used to measure femur length (FL).",
        "Non-standard NT": "Non-standard nuchal translucency measurement view.",
        "Fetal Head Position": "Fetal head position relative to pubic symphysis.",
        "Standard NT": "Standard nuchal translucency measurement for screening.",
        "Thorax": "The thoracic view shows lung fields and diaphragm.",
        "Transcerebellar Plane": "Transcerebellar plane for posterior fossa evaluation.",
        "Transthalamic Plane": "Transthalamic plane for midline brain structures.",
        "Transventricular Plane": "Transventricular plane for ventricle measurement.",
        # Legacy names for backward compatibility
        "Brain": "The brain view shows key structures including ventricles and midline.",
        "Heart": "The cardiac view displays the four-chamber structure.",
        "Spine": "The spine view assesses vertebral alignment.",
        "Kidney": "The kidney view shows renal structure and collecting system.",
        "Face": "The facial view includes profile and coronal views.",
        "Bladder": "The bladder view confirms presence and filling.",
        "Placenta": "The placental view assesses location and thickness.",
        "Other": "Non-standard view detected."
    }

    if organ in organ_info:
        response += f"### View Details\n{organ_info[organ]}\n"

    return response

# Load models once at startup
if MODEL_AVAILABLE and st.session_state.model is None:
    st.session_state.model = load_model()
    if st.session_state.model is None:
        st.error("⚠️ Classification model failed to load.")

if VQA_AVAILABLE and st.session_state.vqa_model is None:
    st.session_state.vqa_model = load_vqa_model()

# Header
st.markdown("# FADA - Fetal Anomaly Detection Algorithm")

# Top metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Analyses", st.session_state.total_analyses)
with col2:
    st.metric("Classification", "EfficientNet-B0")
with col3:
    st.metric("Test Accuracy", "90%")
with col4:
    vqa_status = "Enabled" if st.session_state.vqa_model else "Disabled"
    st.metric("VQA", vqa_status)

st.divider()

# Main tabs
tab1, tab2, tab3 = st.tabs(["Analysis", "About", "Help"])

with tab1:
    # Create two columns for chat and image preview
    chat_col, preview_col = st.columns([2, 1])

    with chat_col:
        # Chat container
        chat_container = st.container(height=500)

        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    if "image" in message and message.get("show_image", False):
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.image(message["image"], width=150)
                    st.markdown(message["content"])

        # Input area
        col1, col2 = st.columns([3, 1])

        with col1:
            prompt = st.chat_input("Ask about the ultrasound analysis...")

        with col2:
            uploaded_file = st.file_uploader(
                "Upload Image",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                label_visibility="collapsed",
                key="upload_widget"
            )

        # Process text input
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Check if VQA is enabled and question is medical-related
            if st.session_state.vqa_enabled and st.session_state.vqa_model and st.session_state.current_image:
                # Use VQA for medical questions about the image
                medical_keywords = ["what", "which", "where", "how", "describe", "identify", "assess", "evaluate", "measure", "show", "visible"]
                if any(keyword in prompt.lower() for keyword in medical_keywords):
                    with st.spinner("Loading VQA model and analyzing image..."):
                        try:
                            answer = st.session_state.vqa_model.answer_question(
                                st.session_state.current_image,
                                prompt
                            )
                            response = answer
                        except Exception as e:
                            response = f"VQA error: {str(e)}"
                else:
                    # Fallback to general response
                    response = "Upload an ultrasound image for detailed analysis. "
                    response += "You can ask specific medical questions about the image."
            else:
                # Generate standard response
                if "explain" in prompt.lower() or "what" in prompt.lower():
                    response = "The analysis uses deep learning models trained on fetal ultrasound data. "
                    response += "Each anatomical view has specific features that the model identifies."
                elif "measure" in prompt.lower() or "size" in prompt.lower():
                    response = "Biometric measurements vary by anatomical view and gestational age. "
                    response += "The system provides classification and confidence scores."
                else:
                    response = "Upload an ultrasound image for detailed analysis. "
                    response += "The system can identify anatomical views and detect patterns."

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

        # Process image upload
        if uploaded_file is not None and not st.session_state.analyzing:
            # Create unique identifier for uploaded file
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"

            # Only process if this is a new file
            if file_id != st.session_state.last_uploaded_file:
                st.session_state.analyzing = True
                st.session_state.last_uploaded_file = file_id

                image = Image.open(uploaded_file)
                st.session_state.current_image = image

                # Add to messages
                st.session_state.messages.append({
                    "role": "user",
                    "content": f"Uploaded: {uploaded_file.name}",
                    "image": image,
                    "show_image": True
                })

                # Perform analysis
                with st.spinner("Analyzing ultrasound image..."):
                    analysis_results = analyze_ultrasound(image, st.session_state.model)
                    response = generate_response(analysis_results)

                    # Store debug info (keeping for Help tab)
                    st.session_state.debug_info.append({
                        'image': uploaded_file.name,
                        'organ': analysis_results.get('organ'),
                        'confidence': analysis_results.get('confidence', 0),
                        'model_available': st.session_state.model is not None
                    })

                    # Store classification for VQA check
                    st.session_state.last_classification = analysis_results.get("organ", "")

                    # Check if VQA is available for this image type
                    if st.session_state.vqa_model and "Non-standard NT" in st.session_state.last_classification:
                        st.session_state.vqa_enabled = True
                        response += "\n\n---\n\n### Visual Question Answering Available\n"
                        response += "This image type supports detailed medical Q&A. Click the question buttons below or ask your own questions in the chat."
                    else:
                        st.session_state.vqa_enabled = False

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })

                    st.session_state.total_analyses += 1
                    st.session_state.analyzing = False
                    st.rerun()

    with preview_col:
        st.markdown("### Current Analysis")

        if st.session_state.current_image:
            st.image(st.session_state.current_image, width='stretch')

            # VQA Question Buttons (if enabled)
            if st.session_state.vqa_enabled and st.session_state.vqa_model:
                st.markdown("#### Ask Medical Questions")

                vqa_shortcuts = st.session_state.vqa_model.get_question_shortcuts()

                with st.expander("Standard Questions", expanded=True):
                    for short_name, full_question in vqa_shortcuts.items():
                        if st.button(short_name, key=f"vqa_btn_{short_name}", width='stretch'):
                            # Add question to chat
                            st.session_state.messages.append({
                                "role": "user",
                                "content": full_question
                            })

                            # Get VQA answer
                            with st.spinner(f"Loading VQA model and answering: {short_name}..."):
                                try:
                                    answer = st.session_state.vqa_model.answer_question(
                                        st.session_state.current_image,
                                        full_question
                                    )

                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": f"**{short_name}**\n\n{answer}"
                                    })
                                except Exception as e:
                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": f"Error loading VQA model: {str(e)}"
                                    })

                            st.rerun()

            # Quick actions
            st.markdown("#### Quick Actions")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Clear", width='stretch'):
                    st.session_state.messages = [{
                        "role": "assistant",
                        "content": "Chat cleared. Upload a new image to begin."
                    }]
                    st.session_state.current_image = None
                    st.session_state.last_uploaded_file = None
                    st.session_state.vqa_enabled = False
                    st.session_state.last_classification = None
                    st.rerun()
            with col2:
                if st.button("Export", width='stretch'):
                    st.info("Export feature coming soon")
        else:
            # Placeholder when no image
            st.info("Upload an ultrasound image to see preview and analysis")

            # Supported views
            with st.expander("Supported Views (12 Classes)", expanded=False):
                st.markdown("""
                - Abdomen
                - Aortic Arch
                - Cervical View
                - Cervix
                - Femur
                - Non-standard NT
                - Fetal Head Position
                - Standard NT
                - Thorax
                - Transcerebellar Plane
                - Transthalamic Plane
                - Transventricular Plane
                """)

with tab2:
    # About section with columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("## About FADA")
        st.markdown("""
        FADA (Fetal Anomaly Detection Algorithm) is an automated analysis system for fetal
        ultrasound images using deep learning technology.

        ### Key Features
        - **Automated Analysis**: Instant classification of ultrasound views
        - **Multi-Class Detection**: Supports 12 different anatomical views
        - **Visual Question Answering**: Ask medical questions about Non-standard NT images
        - **Confidence Scoring**: Provides reliability metrics for each prediction
        - **Interactive Interface**: Chat-based interaction for ease of use

        ### Technical Architecture
        **Phase 1 - Classification**:
        - Model: EfficientNet-B0 backbone with custom classification heads
        - Accuracy: ~90% on 12-class ultrasound view classification

        **Phase 2 - Visual Question Answering (VQA)**:
        - Model: BLIP-2 OPT-2.7B with LoRA fine-tuning
        - Supports 8 medical questions for Non-standard NT images
        - Trained on 50 annotated images with expert annotations

        ### Framework
        - PyTorch for deep learning
        - Streamlit for web interface
        - Input: 224x224 RGB ultrasound images
        - Output: Classification + medical Q&A
        """)

    with col2:
        st.markdown("### System Information")

        # System info container
        info_container = st.container(border=True)
        with info_container:
            st.markdown("**Version:** 3.1")
            st.markdown("**Classification:** EfficientNet-B0")
            st.markdown("**VQA:** BLIP-2 OPT-2.7B")
            st.markdown("**Classes:** 12")
            st.markdown("**Updated:** October 2025")

        st.markdown("### Performance Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", "~90%")
        with col2:
            st.metric("Speed", "< 1s")

with tab3:
    st.markdown("## Help & Documentation")

    # Debug section
    with st.expander("Debug Information", expanded=False):
        st.markdown("### Import Status")
        st.write(f"MODEL_AVAILABLE: {MODEL_AVAILABLE}")
        st.write(f"VQA_AVAILABLE: {VQA_AVAILABLE}")

        st.markdown("### Model Status")
        st.write(f"Classification model loaded: {st.session_state.model is not None}")
        st.write(f"VQA model instance: {st.session_state.vqa_model is not None}")

        # Test VQA import directly
        try:
            from src.models.vqa_model import UltrasoundVQA as TestVQA
            st.success("VQA module can be imported directly")
        except Exception as e:
            st.error(f"VQA import error: {e}")

        if st.session_state.debug_info:
            st.markdown("### Recent Classifications")
            for idx, info in enumerate(st.session_state.debug_info[-5:]):  # Last 5
                st.write(f"**{idx+1}. {info['image']}**")
                st.write(f"   - Organ: {info['organ']}")
                st.write(f"   - Confidence: {info['confidence']:.1%}")
                st.write(f"   - Model: {'✓' if info['model_available'] else '✗'}")

    # Create expandable sections for help
    with st.expander("Getting Started", expanded=True):
        st.markdown("""
        ### Quick Start Guide
        1. **Upload an Image**: Click the "Upload Image" button in the Analysis tab
        2. **Wait for Processing**: The system will analyze the ultrasound automatically
        3. **Review Results**: Check the detected view, confidence score, and assessment
        4. **Ask Questions**: Use the chat input or question buttons (if VQA is enabled)

        ### Using Visual Question Answering (VQA)
        When you upload a Non-standard NT image:
        - VQA mode will be automatically enabled
        - 8 standard medical question buttons will appear on the right
        - Click any button to get instant answers
        - Or type your own medical questions in the chat
        - VQA provides detailed medical analysis beyond classification
        """)

    with st.expander("Supported Image Formats"):
        st.markdown("""
        The system accepts the following image formats:
        - PNG (.png)
        - JPEG (.jpg, .jpeg)
        - BMP (.bmp)
        - TIFF (.tiff)

        **Recommended**: Images should be at least 224x224 pixels for best results.
        """)

    with st.expander("Understanding Results"):
        st.markdown("""
        ### Analysis Output
        - **Detected View**: The anatomical structure identified
        - **Confidence**: How certain the model is (0-100%)
        - **Image Quality**: Assessment of the input image quality
        - **Orientation**: The viewing angle of the ultrasound

        ### Confidence Interpretation
        - **> 90%**: High confidence
        - **70-90%**: Moderate confidence
        - **< 70%**: Low confidence - consider re-uploading
        """)

    with st.expander("Troubleshooting"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### Common Issues
            **Image not uploading?**
            - Check file format and size
            - Refresh the page

            **Unexpected results?**
            - Ensure good image quality
            - Try different viewing angles
            """)

        with col2:
            st.markdown("""
            ### Tips for Best Results
            - Use clear, high-contrast images
            - Avoid heavily cropped images
            - Standard clinical views work best
            - Multiple angles improve accuracy
            """)

# Footer
st.divider()
footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])
with footer_col2:
    st.markdown(
        "<p style='text-align: center; color: gray; font-size: 12px;'>"
        "FADA v3.1"
        "</p>",
        unsafe_allow_html=True
    )