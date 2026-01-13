"""Test pre-quantized Qwen-VL-Chat-Int4 version"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from pathlib import Path
import time

print("="*70)
print("Testing Qwen-VL-Chat-Int4 (Pre-quantized version)")
print("="*70)

# Check CUDA
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

DATA_DIR = "data/Fetal Ultrasound"

# Try pre-quantized Int4 version
MODEL_ID = "Qwen/Qwen-VL-Chat-Int4"

try:
    print(f"\n1. Loading pre-quantized {MODEL_ID}...")
    print("   This should be faster than quantizing on-the-fly...")
    start_time = time.time()

    # Load tokenizer
    print("   Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Load pre-quantized model
    print("   Loading Int4 model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        device_map="auto"
    ).eval()

    load_time = time.time() - start_time
    print(f"   [OK] Model loaded in {load_time:.1f} seconds!")

    # Get model info
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"   GPU Memory used: {memory:.2f} GB")

    print("\n2. Quick test on ultrasound image...")

    # Get one test image
    test_image = None
    for cat_dir in list(Path(DATA_DIR).iterdir())[:1]:
        if cat_dir.is_dir():
            images = list(cat_dir.glob("*.png"))[:1]
            if images:
                test_image = images[0]
                break

    if test_image:
        print(f"   Image: {test_image.name}")
        print(f"   Category: {test_image.parent.name}")

        # Load and save image
        image = Image.open(test_image).convert('RGB')
        temp_path = "temp_qwen_test.png"
        image.save(temp_path)

        # Create query
        query = tokenizer.from_list_format([
            {'image': temp_path},
            {'text': 'What anatomical structures can you see in this ultrasound image?'},
        ])

        # Generate
        print("   Generating response...")
        start = time.time()
        response, _ = model.chat(tokenizer, query=query, history=None)
        gen_time = time.time() - start

        print(f"   Response: {response}")
        print(f"   Time: {gen_time:.2f}s")

        # Clean up
        Path(temp_path).unlink(missing_ok=True)

    print("\n" + "="*70)
    print("SUCCESS! Qwen-VL-Chat-Int4 works!")
    print("="*70)

except Exception as e:
    print(f"\n[ERROR] {type(e).__name__}: {str(e)[:200]}")
    print("\nQwen-VL-Chat-Int4 may not be available or compatible")

print("\n" + "="*70)