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
FULL_TEST_DATA = DATA_DIR / "vlm_training" / "gemini_complete_test.jsonl"
STRATIFIED_TEST_DATA = OUTPUTS_DIR / "test_subset.jsonl"

# Categories (12 total from dataset - note "Abodomen" is the actual folder name with typo)
CATEGORIES = [
    "Abodomen",  # Note: folder has typo
    "Aorta",
    "Cervical",
    "Cervix",
    "Femur",
    "Non_standard_NT",
    "Standard_NT",
    "Thorax",
    "Trans-cerebellum",
    "Trans-thalamic",
    "Trans-ventricular",
    "Public_Symphysis_fetal_head"
]

# Evaluation settings
DEFAULT_SAMPLES_PER_CATEGORY = 50
DEFAULT_EMBEDDING_MODEL = "all-mpnet-base-v2"
FALLBACK_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Generation settings
SYSTEM_PROMPT = """You are an expert in fetal ultrasound imaging analysis. Provide accurate, detailed, and clinically relevant interpretations. Be precise and professional in your assessments."""

MAX_NEW_TOKENS = 1024
GENERATION_TEMPERATURE = 0.1  # Low for consistency

# Standard questions
QUESTIONS = [
    "Anatomical Structures Identification: Identify and describe all anatomical structures visible in the image.",
    "Fetal Orientation: Determine the orientation of the fetus based on the image (e.g., head up/down, front/back view).",
    "Plane Evaluation: Assess if the image is taken at a standard diagnostic plane and describe its diagnostic relevance.",
    "Biometric Measurements: Identify any measurable biometric parameters (e.g., femur length, head circumference) from the image.",
    "Gestational Age: Estimate the gestational age of the fetus based on the visible features.",
    "Image Quality: Assess the quality of the ultrasound image, mentioning any factors that might affect its interpretation (e.g., clarity, artifacts).",
    "Normality / Abnormality: Determine whether the observed structures appear normal or identify any visible abnormalities or concerns.",
    "Clinical Recommendations: Provide any relevant clinical recommendations or suggested next steps based on your interpretation."
]

QUESTION_SHORT_NAMES = [
    "Anatomical Structures",
    "Fetal Orientation",
    "Plane Evaluation",
    "Biometric Measurements",
    "Gestational Age",
    "Image Quality",
    "Normality/Abnormality",
    "Clinical Recommendations"
]
