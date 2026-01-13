"""Test Qwen-VL vision-language model with quantization"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from PIL import Image
from pathlib import Path
import time

print("="*70)
print("Qwen-VL Test with Quantization")
print("="*70)

# Check CUDA
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

DATA_DIR = "data/Fetal Ultrasound"
MODEL_ID = "Qwen/Qwen-VL"  # Original Qwen-VL model

try:
    print(f"\n1. Loading Qwen-VL with 4-bit quantization...")
    print(f"   Model: {MODEL_ID}")

    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        trust_remote_code=True,
        device_map="auto"
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"   [OK] Model loaded: {param_count / 1e9:.2f}B params")

    if torch.cuda.is_available():
        memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"   GPU Memory: {memory:.2f} GB")

    print("\n2. Testing on sample images...")

    # Get test images
    test_images = []
    for cat_dir in list(Path(DATA_DIR).iterdir())[:3]:
        if cat_dir.is_dir():
            images = list(cat_dir.glob("*.png"))[:1]
            if images:
                test_images.append(images[0])

    for i, img_path in enumerate(test_images):
        print(f"\n   Image {i+1}: {img_path.name}")
        print(f"   Category: {img_path.parent.name}")

        image = Image.open(img_path).convert('RGB')

        # Qwen-VL uses special format for image inputs
        query = tokenizer.from_list_format([
            {'image': str(img_path)},  # Qwen-VL expects path
            {'text': 'What anatomical structures can you see in this ultrasound image?'},
        ])

        # Generate response
        start = time.time()
        response, _ = model.chat(
            tokenizer,
            query=query,
            history=None
        )
        gen_time = time.time() - start

        print(f"   Response: {response}")
        print(f"   Time: {gen_time:.2f}s")

    print("\n" + "="*70)
    print("SUCCESS! Qwen-VL works with 4-bit quantization")
    print("="*70)

except Exception as e:
    print(f"\n[ERROR] {type(e).__name__}: {e}")
    print("\nQwen-VL may require special installation or different API")
    print("Checking for Qwen-VL-Chat variant...")

    # Try Qwen-VL-Chat variant
    try:
        MODEL_ID = "Qwen/Qwen-VL-Chat"
        print(f"\nTrying: {MODEL_ID}")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quantization_config,
            trust_remote_code=True,
            device_map="auto"
        )

        print(f"Qwen-VL-Chat loaded successfully!")

    except Exception as e2:
        print(f"Also failed: {e2}")
        print("\nQwen-VL models may require:")
        print("1. Special installation: pip install qwen-vl")
        print("2. Different API than standard transformers")
        print("3. Model size may exceed 8GB even with quantization")

print("\n" + "="*70)