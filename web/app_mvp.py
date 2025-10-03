"""
FADA - Fetal Anomaly Detection Algorithm
Multi-Model VLM Comparison Interface (MVP)

This MVP demonstrates the multi-model VQA comparison workflow:
1. User uploads ultrasound image
2. System classifies organ type
3. Top-3 VLM models answer 8 standard questions
4. Sonographer selects best answer for each question
5. Selections saved for analysis

Design: Modular architecture allows swapping local GPU inference with API endpoints
"""

import streamlit as st
import sys
from pathlib import Path
from PIL import Image
import pandas as pd
import json
from datetime import datetime
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.question_loader import get_question_loader
from src.inference.vlm_interface import create_top_vlms, VLMManager

# Page config
st.set_page_config(
    page_title="FADA - Multi-Model VQA Comparison",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better comparison layout
st.markdown("""
<style>
.stRadio > div {flex-direction: row;}
.model-card {
    border: 1px solid #ddd;
    border-radius: 5px;
    padding: 10px;
    margin: 5px;
    background-color: #f9f9f9;
}
.selected-model {
    background-color: #e8f4f8;
    border: 2px solid #1f77b4;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "vlm_manager" not in st.session_state:
    st.session_state.vlm_manager = None

if "use_api" not in st.session_state:
    st.session_state.use_api = False

if "current_image" not in st.session_state:
    st.session_state.current_image = None

if "detected_type" not in st.session_state:
    st.session_state.detected_type = None

if "model_responses" not in st.session_state:
    st.session_state.model_responses = {}

if "user_selections" not in st.session_state:
    st.session_state.user_selections = {}

if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False


def init_vlm_manager():
    """Initialize VLM manager based on configuration"""
    if st.session_state.vlm_manager is not None:
        return st.session_state.vlm_manager

    with st.spinner("Initializing VLM models..."):
        # For MVP: use local GPU inference
        # TODO: Add API option in sidebar settings
        manager = create_top_vlms(use_api=st.session_state.use_api)
        st.session_state.vlm_manager = manager
        return manager


def classify_image(image: Image.Image) -> str:
    """
    Classify image to detect organ type

    For MVP: Simple heuristic or placeholder
    TODO: Integrate actual classification model

    Args:
        image: PIL Image

    Returns:
        Detected category (e.g., "Brain", "Heart", "Abdomen")
    """
    # Placeholder: In real implementation, use trained classifier
    # For now, ask user or use default
    return "Unknown"


def run_vlm_analysis(image: Image.Image, questions: list, models: list[str]):
    """
    Run all VLM models on image with questions

    Args:
        image: PIL Image
        questions: List of question texts
        models: List of model keys to run

    Returns:
        Dict mapping model_key -> list of answers
    """
    manager = init_vlm_manager()
    responses = {}

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, model_key in enumerate(models):
        model = manager.get_model(model_key)
        status_text.text(f"Loading {model.model_name}...")

        # Load model
        model.load()

        status_text.text(f"Analyzing with {model.model_name}...")

        # Get answers
        try:
            answers = model.answer_batch(image, questions)
            responses[model_key] = answers
        except Exception as e:
            st.error(f"Error with {model.model_name}: {e}")
            responses[model_key] = [f"Error: {str(e)}"] * len(questions)

        # Unload model to free memory
        model.unload()

        # Update progress
        progress = (idx + 1) / len(models)
        progress_bar.progress(progress)

    progress_bar.empty()
    status_text.empty()

    return responses


def save_user_selections(image_name: str, detected_type: str, selections: dict):
    """
    Save user selections to JSON file

    Args:
        image_name: Name of analyzed image
        detected_type: Detected organ type
        selections: Dict mapping question_idx -> selected_model_key
    """
    output_dir = Path("outputs/user_selections")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"selection_{timestamp}.json"

    data = {
        "timestamp": timestamp,
        "image_name": image_name,
        "detected_type": detected_type,
        "selections": selections
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    return filename


# ========== UI ==========

st.title("🔬 FADA - Multi-Model VQA Comparison")
st.markdown("**Research Prototype** - Not for clinical use")

st.markdown("""
### Workflow
1. Upload fetal ultrasound image
2. System detects organ type (automatic)
3. Top-3 VLM models answer 8 standard questions
4. Select best answer for each question
5. Submit selections for analysis
""")

# Sidebar settings
with st.sidebar:
    st.header("Settings")

    st.session_state.use_api = st.checkbox(
        "Use API Endpoints",
        value=st.session_state.use_api,
        help="Use cloud API instead of local GPU (requires endpoint configuration)"
    )

    if st.session_state.use_api:
        api_endpoint = st.text_input("API Endpoint", value="http://localhost:8000")
        api_key = st.text_input("API Key", type="password")
    else:
        st.info("Using local GPU inference")

    st.markdown("---")
    st.markdown("### Models")
    st.markdown("- MiniCPM-V-2.6 (88.9%)")
    st.markdown("- Qwen2-VL-2B (83.3%)")
    st.markdown("- InternVL2-4B (82%)")

# Main interface
col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Upload Image")

    uploaded_file = st.file_uploader(
        "Choose ultrasound image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload fetal ultrasound image for analysis"
    )

    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file).convert('RGB')
        st.session_state.current_image = image

        # Display image
        st.image(image, caption=uploaded_file.name, use_container_width=True)

        # Classify image
        if st.session_state.detected_type is None:
            st.session_state.detected_type = classify_image(image)

        # Manual type override
        st.markdown("**Detected Type:**")
        organ_types = ["Brain", "Heart", "Abdomen", "Femur", "Thorax", "Cervix", "Aorta", "Other"]
        selected_type = st.selectbox(
            "Confirm or correct organ type",
            options=organ_types,
            index=organ_types.index(st.session_state.detected_type) if st.session_state.detected_type in organ_types else 0
        )
        st.session_state.detected_type = selected_type

        # Run analysis button
        if st.button("🚀 Run Multi-Model Analysis", type="primary", use_container_width=True):
            with st.spinner("Running analysis on 3 models..."):
                # Load questions
                question_loader = get_question_loader()
                questions = question_loader.get_questions()

                # Run models
                responses = run_vlm_analysis(
                    image,
                    questions,
                    models=["minicpm", "qwen2vl", "internvl2"]
                )

                st.session_state.model_responses = responses
                st.session_state.analysis_complete = True
                st.success("✓ Analysis complete!")
                st.rerun()

with col2:
    st.header("2. Compare Model Responses")

    if not st.session_state.analysis_complete:
        st.info("👈 Upload an image and run analysis to see results")
    else:
        # Load questions
        question_loader = get_question_loader()
        questions = question_loader.get_questions()
        short_names = question_loader.get_question_short_names()

        # Display comparison for each question
        st.markdown(f"**Image**: {uploaded_file.name if uploaded_file else 'N/A'}")
        st.markdown(f"**Type**: {st.session_state.detected_type}")
        st.markdown("---")

        # Initialize selections if empty
        if not st.session_state.user_selections:
            st.session_state.user_selections = {i: None for i in range(len(questions))}

        for q_idx, (short_name, question) in enumerate(zip(short_names, questions)):
            st.markdown(f"### {short_name}")
            st.markdown(f"*{question}*")

            # Create columns for each model
            cols = st.columns(3)

            model_keys = ["minicpm", "qwen2vl", "internvl2"]
            model_names = ["MiniCPM-V-2.6", "Qwen2-VL-2B", "InternVL2-4B"]

            for col, model_key, model_name in zip(cols, model_keys, model_names):
                with col:
                    # Get answer
                    answer = st.session_state.model_responses.get(model_key, ["N/A"] * len(questions))[q_idx]

                    # Display model card
                    st.markdown(f"**{model_name}**")
                    st.markdown(f'<div class="model-card">{answer}</div>', unsafe_allow_html=True)

            # Selection radio buttons
            selection = st.radio(
                f"Select best answer for {short_name}",
                options=model_keys,
                format_func=lambda x: {"minicpm": "MiniCPM-V-2.6", "qwen2vl": "Qwen2-VL-2B", "internvl2": "InternVL2-4B"}[x],
                key=f"q{q_idx}_selection",
                horizontal=True,
                index=model_keys.index(st.session_state.user_selections[q_idx]) if st.session_state.user_selections[q_idx] in model_keys else 0
            )

            st.session_state.user_selections[q_idx] = selection

            st.markdown("---")

        # Submit button
        if st.button("📊 Submit Selections", type="primary", use_container_width=True):
            # Save selections
            filename = save_user_selections(
                image_name=uploaded_file.name if uploaded_file else "unknown",
                detected_type=st.session_state.detected_type,
                selections=st.session_state.user_selections
            )

            st.success(f"✓ Selections saved to: {filename}")

            # Show summary
            st.markdown("### Selection Summary")
            selection_counts = {}
            for model_key in ["minicpm", "qwen2vl", "internvl2"]:
                count = sum(1 for v in st.session_state.user_selections.values() if v == model_key)
                selection_counts[model_key] = count

            summary_df = pd.DataFrame([
                {"Model": "MiniCPM-V-2.6", "Selected": selection_counts.get("minicpm", 0)},
                {"Model": "Qwen2-VL-2B", "Selected": selection_counts.get("qwen2vl", 0)},
                {"Model": "InternVL2-4B", "Selected": selection_counts.get("internvl2", 0)},
            ])

            st.dataframe(summary_df, use_container_width=True)

            # Reset button
            if st.button("🔄 Analyze Another Image"):
                st.session_state.current_image = None
                st.session_state.detected_type = None
                st.session_state.model_responses = {}
                st.session_state.user_selections = {}
                st.session_state.analysis_complete = False
                st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
<p><strong>FADA - Fetal Anomaly Detection Algorithm</strong></p>
<p>Research Prototype • Not for Clinical Use • Multi-Model VQA Comparison MVP</p>
<p>Models: MiniCPM-V-2.6 (88.9%) • Qwen2-VL-2B (83.3%) • InternVL2-4B (82%)</p>
</div>
""", unsafe_allow_html=True)
