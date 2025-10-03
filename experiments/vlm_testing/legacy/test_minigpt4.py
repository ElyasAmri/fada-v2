"""Test MiniGPT-4 vision-language model"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from PIL import Image
from pathlib import Path
import time

print("="*70)
print("MiniGPT-4 Test")
print("="*70)

# Check CUDA
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

DATA_DIR = "data/Fetal Ultrasound"

# Try different MiniGPT-4 model IDs
model_ids = [
    "Vision-CAIR/MiniGPT-4",
    "mlpc-lab/MiniGPT-4",
    "BAAI/MiniGPT-4",
    "miniGPT-4/miniGPT-4",
]

for MODEL_ID in model_ids:
    print(f"\nTrying model: {MODEL_ID}")
    try:
        # Try with quantization for 8GB GPU
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        print(f"Loading model with 4-bit quantization...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quantization_config,
            trust_remote_code=True,
            device_map="auto"
        )

        print(f"Model loaded successfully!")
        break

    except Exception as e:
        print(f"Failed: {e}")
        continue
else:
    print("\nNo valid MiniGPT-4 model found. Checking alternative approaches...")

    # MiniGPT-4 might require special installation
    print("\nMiniGPT-4 typically requires:")
    print("1. Clone from: https://github.com/Vision-CAIR/MiniGPT-4")
    print("2. Custom installation process")
    print("3. Download pretrained checkpoints separately")
    print("\nThis is a complex setup model - marking as SKIP")