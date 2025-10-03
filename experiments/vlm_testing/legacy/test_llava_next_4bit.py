"""Test LLaVA-NeXT-7B with 4-bit quantization on FADA dataset"""

import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
from pathlib import Path
import time

print("="*70)
print("LLaVA-NeXT-7B (4-bit) Quick Test on FADA Dataset")
print("="*70)

# Configuration
MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"  # LLaVA-NeXT with Mistral-7B
DATA_DIR = "data/Fetal Ultrasound"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n1. Loading LLaVA-NeXT-7B with 4-bit quantization...")
print(f"   Model: {MODEL_ID}")
print(f"   Device: {device}")
print(f"   Quantization: 4-bit NF4 (bitsandbytes)")

start = time.time()

try:
    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True  # Nested quantization for more savings
    )

    # Load processor and model
    processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto"
    )

    load_time = time.time() - start
    print(f"   [OK] Model loaded in {load_time:.1f} seconds")

    # Get model size
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {param_count / 1e9:.2f}B")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"   GPU Memory: {memory:.2f} GB")

    print("\n2. Testing VQA on sample images...")

    # Collect test images
    test_images = []
    category_dirs = [d for d in Path(DATA_DIR).iterdir() if d.is_dir()]

    for cat_dir in category_dirs[:5]:
        cat_name = cat_dir.name
        images = list(cat_dir.glob("*.png")) + list(cat_dir.glob("*.jpg"))
        if not images:
            continue
        for img_path in images[:3]:
            test_images.append({'path': img_path, 'category': cat_name})

    print(f"   Testing {len(test_images)} images...")

    questions = [
        "What anatomical structures can you see in this ultrasound image?",
        "Describe this fetal ultrasound image.",
        "What part of the fetus is shown in this image?"
    ]

    results = []

    for i, item in enumerate(test_images[:5]):
        img_path = item['path']
        true_cat = item['category']

        print(f"\n   Image {i+1}/{5}: {img_path.name}")
        print(f"   True category: {true_cat}")

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Prepare prompt
        prompt = f"[INST] <image>\n{questions[0]} [/INST]"

        # Process inputs - pass image and text separately
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

        # Generate response
        start_gen = time.time()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False
            )
        gen_time = time.time() - start_gen

        # Decode response
        response = processor.decode(output[0], skip_special_tokens=True)
        # Extract just the response part
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()

        print(f"   Question: {questions[0]}")
        print(f"   Response: {response}")
        print(f"   Generation time: {gen_time:.2f}s")

        results.append({
            'image': img_path.name,
            'category': true_cat,
            'question': questions[0],
            'response': response,
            'time': gen_time
        })

    # Summary
    print("\n" + "="*70)
    print("Quick Test Summary")
    print("="*70)
    print(f"\nModel: LLaVA-NeXT-7B (4-bit quantized)")
    print(f"Parameters: {param_count / 1e9:.2f}B")
    print(f"Images tested: {len(results)}")

    avg_time = sum(r['time'] for r in results) / len(results)
    print(f"Average generation time: {avg_time:.2f}s")

    if torch.cuda.is_available():
        final_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"GPU Memory used: {final_memory:.2f} GB")

    print("\n" + "="*70)
    print("Comparison with Other Models")
    print("="*70)
    print(f"\nLLaVA-NeXT-7B-4bit: {param_count / 1e9:.2f}B params, ~{final_memory if torch.cuda.is_available() else 'N/A':.1f}GB memory")
    print(f"InstructBLIP-7B-4b: 4.10B params, ~5.2GB memory")
    print(f"BLIP-2:             3.4B params, ~4.2GB memory")
    print(f"PaliGemma-8bit:     2.92B params, ~3.7GB memory")

    print("\nNote: LLaVA-NeXT improves resolution and reasoning over LLaVA-1.5")
    print("Based on Mistral-7B language model")

    # Save results
    import json
    with open("llava_next_4bit_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: llava_next_4bit_results.json")

except Exception as e:
    print(f"\n[ERROR] {type(e).__name__}: {e}")
    print("\nPossible issues:")
    print("  - bitsandbytes not installed or outdated")
    print("  - Insufficient GPU memory even with 4-bit")
    print("  - Need transformers>=4.36.0 for LLaVA-NeXT")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)