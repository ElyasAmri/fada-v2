"""
Fetal Ultrasound Analysis Chatbot - Streamlit Web Interface
Research prototype for educational purposes only
"""

import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
import sys
import time
import os
from typing import Optional, List, Dict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.chatbot.chatbot import UltrasoundChatbot, AnalysisResult
from src.chatbot.response_generator import ResponseGenerator, ClassificationResult
from src.chatbot.model_config import get_model_config, AVAILABLE_MODELS

# Page configuration
st.set_page_config(
    page_title="Fetal Ultrasound Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ChatGPT-like styling
st.markdown("""
<style>
    /* Remove default padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #202123;
        min-width: 260px;
    }

    [data-testid="stSidebar"] .block-container {
        padding: 1rem;
    }

    /* Navigation button styling */
    .nav-button {
        display: flex;
        align-items: center;
        width: 100%;
        padding: 12px 16px;
        margin: 4px 0;
        border: 1px solid transparent;
        border-radius: 8px;
        background: transparent;
        color: white;
        cursor: pointer;
        transition: background-color 0.2s;
        text-align: left;
        font-size: 14px;
    }

    .nav-button:hover {
        background-color: #343541;
    }

    .nav-button.active {
        background-color: #343541;
        border-color: #565869;
    }

    .nav-icon {
        margin-right: 12px;
        width: 20px;
        text-align: center;
    }

    /* Main content styling */
    .main-header {
        border-bottom: 1px solid #e5e5e5;
        padding-bottom: 1rem;
        margin-bottom: 2rem;
    }

    /* Chat message styling */
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }

    .assistant-message {
        background-color: #f7f7f8;
        border: 1px solid #e5e5e5;
    }

    .user-message {
        background-color: #fff;
        border: 1px solid #e5e5e5;
    }

    /* Confidence indicators */
    .confidence-high {
        color: #10a37f;
        font-weight: bold;
    }

    .confidence-medium {
        color: #ff9800;
        font-weight: bold;
    }

    .confidence-low {
        color: #ef4444;
        font-weight: bold;
    }

    /* Disclaimer styling */
    .disclaimer {
        background-color: #fef3c7;
        border: 1px solid #fbbf24;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar text color fix */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown {
        color: #ececf1 !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        border-bottom: 1px solid #e5e5e5;
    }

    .stTabs [data-baseweb="tab"] {
        height: 44px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px 8px 0 0;
        padding-left: 20px;
        padding-right: 20px;
        color: #666;
        font-size: 14px;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: #fff;
        color: #202123;
        border: 1px solid #e5e5e5;
        border-bottom: 1px solid #fff;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_chatbot():
    """Load and cache the chatbot model"""
    model_path = Path("models/best_model_efficientnet_b0_12class.pth")

    if not model_path.exists():
        st.error(f"Model not found at {model_path}. Please train the model first.")
        return None

    try:
        chatbot = UltrasoundChatbot(
            model_path=str(model_path),
            use_openai=True,
            use_gpu=True,
            confidence_threshold=0.7
        )
        return chatbot
    except Exception as e:
        st.error(f"Failed to load chatbot: {e}")
        return None


def format_confidence(confidence: float) -> str:
    """Format confidence with appropriate styling"""
    if confidence >= 70:
        return f'<span class="confidence-high">{confidence:.1f}%</span>'
    elif confidence >= 50:
        return f'<span class="confidence-medium">{confidence:.1f}%</span>'
    else:
        return f'<span class="confidence-low">{confidence:.1f}%</span>'


def display_analysis_result(result: AnalysisResult):
    """Display analysis result in a formatted way"""
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Classification Results")
        st.markdown(f"**Detected View:** {result.predicted_class}")
        st.markdown(f"**Confidence:** {format_confidence(result.confidence)}", unsafe_allow_html=True)

        st.markdown("**Top 3 Predictions:**")
        for class_name, conf in result.top_3_predictions:
            st.progress(conf/100)
            st.caption(f"{class_name}: {conf:.1f}%")

        st.caption(f"Processing time: {result.processing_time:.2f} seconds")

    with col2:
        st.subheader("Analysis Response")
        st.markdown(f'<div class="chat-message assistant-message">{result.response_text}</div>',
                   unsafe_allow_html=True)


def render_chat_interface():
    """Render the main chat interface"""
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("Ultrasound Analysis")
    st.markdown('</div>', unsafe_allow_html=True)

    # Warning banner
    st.warning("""
    **Research Prototype:** For educational purposes only. Not for clinical use.
    Always consult healthcare professionals for medical interpretation.
    """)

    # Upload section
    with st.container():
        uploaded_file = st.file_uploader(
            "Upload an ultrasound image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a fetal ultrasound image for analysis"
        )

        user_question = st.text_input(
            "Ask a specific question (optional)",
            placeholder="e.g., What measurements can be taken from this view?"
        )

        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            analyze_btn = st.button("Analyze", type="primary", disabled=uploaded_file is None)

        if analyze_btn and uploaded_file:
            with st.spinner("Analyzing..."):
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_column_width=True)

                    result = st.session_state.chatbot.analyze_image(
                        image,
                        user_question=user_question if user_question else None,
                        include_details=True
                    )

                    st.session_state.analysis_results.append(result)
                    display_analysis_result(result)

                except Exception as e:
                    st.error(f"Error: {e}")

    # Follow-up section
    if st.session_state.analysis_results:
        st.divider()
        with st.container():
            st.subheader("Follow-up Questions")

            followup = st.text_input(
                "Ask a follow-up question",
                placeholder="e.g., When is this scan typically performed?"
            )

            if st.button("Ask", disabled=not followup):
                with st.spinner("Generating response..."):
                    response = st.session_state.chatbot.answer_followup(followup)
                    st.markdown(f'<div class="chat-message assistant-message">{response}</div>',
                               unsafe_allow_html=True)


def render_batch_interface():
    """Render batch analysis interface"""
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("Batch Analysis")
    st.markdown('</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload multiple ultrasound images",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        key="batch_uploader"
    )

    if st.button("Analyze All", type="primary", disabled=not uploaded_files):
        progress_bar = st.progress(0)
        all_results = []

        for i, file in enumerate(uploaded_files):
            progress_bar.progress((i + 1) / len(uploaded_files))

            with st.spinner(f"Analyzing {i+1}/{len(uploaded_files)}..."):
                try:
                    image = Image.open(file)
                    result = st.session_state.chatbot.analyze_image(image, include_details=False)
                    all_results.append(result)
                except Exception as e:
                    st.error(f"Error with {file.name}: {e}")

        if all_results:
            st.success(f"Analyzed {len(all_results)} images")

            # Statistics
            class_counts = {}
            avg_confidence = sum(r.confidence for r in all_results) / len(all_results)

            for result in all_results:
                class_counts[result.predicted_class] = class_counts.get(result.predicted_class, 0) + 1

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average Confidence", f"{avg_confidence:.1f}%")
            with col2:
                st.metric("Images Analyzed", len(all_results))

            st.subheader("Distribution")
            for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                st.write(f"• **{class_name}**: {count} image(s)")


def render_settings():
    """Render settings interface"""
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("Settings")
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Analysis Settings")

        new_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.5,
            max_value=0.9,
            value=st.session_state.confidence_threshold,
            step=0.05,
            help="Minimum confidence for definitive statements"
        )

        if new_threshold != st.session_state.confidence_threshold:
            st.session_state.confidence_threshold = new_threshold
            if st.session_state.chatbot:
                st.session_state.chatbot.confidence_threshold = new_threshold

        st.info(f"Current: {st.session_state.confidence_threshold:.0%}")

    with col2:
        st.subheader("OpenAI Integration")

        st.session_state.use_openai = st.checkbox(
            "Enable GPT Responses",
            value=st.session_state.use_openai
        )

        if st.session_state.use_openai:
            current_model = os.getenv("MODEL_NAME", "gpt-4o-mini")
            model_config = get_model_config(current_model)

            if model_config:
                st.success(f"Model: {model_config.display_name}")
                st.caption(f"{model_config.speed} speed • {model_config.cost_tier} cost")

    st.divider()

    st.subheader("Model Information")
    col3, col4 = st.columns(2)
    with col3:
        st.metric("Architecture", "EfficientNet-B0")
        st.metric("Classes", "12 views")
    with col4:
        st.metric("Accuracy", "90.14%")
        st.metric("Training Images", "15,002")

    st.divider()

    if st.button("Clear History", type="secondary"):
        st.session_state.conversation_history = []
        st.session_state.analysis_results = []
        if st.session_state.chatbot:
            st.session_state.chatbot.clear_history()
        st.success("History cleared")
        st.rerun()


def main():
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'chat'
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = []
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'confidence_threshold' not in st.session_state:
        st.session_state.confidence_threshold = 0.7
    if 'use_openai' not in st.session_state:
        st.session_state.use_openai = True

    # Load chatbot
    if st.session_state.chatbot is None:
        with st.spinner("Loading model..."):
            st.session_state.chatbot = load_chatbot()

    if st.session_state.chatbot is None:
        st.error("Failed to load model. Check model file.")
        return

    # Sidebar Navigation
    with st.sidebar:
        st.markdown("### FADA AI")
        st.markdown("---")

        # Navigation buttons (ChatGPT-style)
        if st.button("New Analysis", use_container_width=True, type="primary" if st.session_state.page == 'chat' else "secondary"):
            st.session_state.page = 'chat'
            st.session_state.analysis_results = []
            st.rerun()

        if st.button("Batch Process", use_container_width=True, type="primary" if st.session_state.page == 'batch' else "secondary"):
            st.session_state.page = 'batch'
            st.rerun()

        if st.button("Settings", use_container_width=True, type="primary" if st.session_state.page == 'settings' else "secondary"):
            st.session_state.page = 'settings'
            st.rerun()

        st.markdown("---")

        # Recent analyses (like ChatGPT history)
        if st.session_state.analysis_results:
            st.markdown("### Recent")
            for i, result in enumerate(st.session_state.analysis_results[-5:]):
                if st.button(
                    f"{result.predicted_class[:20]}... ({result.confidence:.0f}%)",
                    key=f"history_{i}",
                    use_container_width=True
                ):
                    st.session_state.page = 'chat'
                    st.rerun()

        # Bottom section
        st.markdown("---")
        with st.expander("About"):
            st.markdown("""
            **FADA Ultrasound AI**

            Version 1.0
            Research Prototype

            **NOT for clinical use**
            """)

            st.caption("Model: EfficientNet-B0")
            st.caption("Accuracy: 90.14%")

        # Status indicators
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Model: Ready")
        with col2:
            if st.session_state.use_openai:
                st.caption("GPT: Active")
            else:
                st.caption("GPT: Off")

    # Main content area
    if st.session_state.page == 'chat':
        render_chat_interface()
    elif st.session_state.page == 'batch':
        render_batch_interface()
    elif st.session_state.page == 'settings':
        render_settings()


if __name__ == "__main__":
    main()