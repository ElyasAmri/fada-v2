"""
Configuration constants for VLM evaluation pipeline.
"""

from pathlib import Path


# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "evaluation"

# Model configuration
BASE_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
ADAPTER_PATH = MODELS_DIR / "qwen25vl7b_finetuned" / "final"

# Test data
FULL_TEST_DATA = DATA_DIR / "vlm_training" / "gt_test.jsonl"
STRATIFIED_TEST_DATA = OUTPUTS_DIR / "test_subset.jsonl"

# Categories (14 total from dataset, sorted alphabetically)
CATEGORIES = [
    "Abdomen",
    "Aorta",
    "CRL-View",
    "Cervical",
    "Cervix",
    "Femur",
    "NT-View",
    "Non_standard_NT",
    "Public_Symphysis_fetal_head",
    "Standard_NT",
    "Thorax",
    "Trans-cerebellum",
    "Trans-thalamic",
    "Trans-ventricular",
]

# Evaluation settings
DEFAULT_SAMPLES_PER_CATEGORY = 50
DEFAULT_EMBEDDING_MODEL = "all-mpnet-base-v2"
FALLBACK_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Generation settings
# Temperature policy: T=0.1 for evaluation (consistency), T=0.7 for interactive inference
VLM_SYSTEM_PROMPT = """You are an expert in fetal ultrasound imaging analysis. Provide accurate, detailed, and clinically relevant interpretations. Be precise and professional in your assessments."""

# Backward-compatible alias
SYSTEM_PROMPT = VLM_SYSTEM_PROMPT

# System prompt for API-based evaluation (Gemini, GPT-4o, Grok)
API_SYSTEM_PROMPT = """You are a medical imaging expert analyzing fetal ultrasound images. Provide clear, professional medical responses."""

MAX_NEW_TOKENS = 1024
GENERATION_TEMPERATURE = 0.1  # Low for consistency

# Ground truth annotations (sonographer, normalized)
ANNOTATIONS_PATH = DATA_DIR / "Fetal Ultrasound Annotations Normalized.xlsx"

# BERTScore model for Q8 evaluation
BERTSCORE_MODEL = "roberta-large"

# Gestational age bins (ordered for adjacency scoring)
GA_BINS_ORDERED = [
    "8-13 weeks", "13-15 weeks", "15-20 weeks", "20-25 weeks",
    "25-30 weeks", "30-35 weeks", "35-38 weeks", "38+ weeks",
]

# Image quality tiers (ordered for adjacency scoring)
QUALITY_TIERS = {"good": 2, "medium": 1, "low": 0}
