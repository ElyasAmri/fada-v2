"""Quick test of InstructBLIP-Vicuna-7B with 4-bit quantization on FADA dataset"""

import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
from pathlib import Path
import time

print("="*70)
print("InstructBLIP-Vicuna-7B (4-bit) Quick Test on FADA Dataset")
print("="*70)

# Configuration
MODEL_ID = "Salesforce/instructblip-vicuna-7b"
DATA_DIR = "data/Fetal Ultrasound"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n1. Loading InstructBLIP-Vicuna-7B with 4-bit quantization...")
print(f"   Model: {MODEL_ID}")
print(f"   Device: {device}")
print(f"   Quantization: 4-bit NF4 (bitsandbytes)")

start = time.time()

try:
    # Configure 4-bit quantization with NF4 (Normal Float 4)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # Use NF4 quantization
        bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bfloat16
        bnb_4bit_use_double_quant=True  # Use nested quantization for more memory savings
    )

    # Load processor and model
    processor = InstructBlipProcessor.from_pretrained(MODEL_ID)
    model = InstructBlipForConditionalGeneration.from_pretrained(
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
        raw_image = Image.open(img_path).convert('RGB')

        # Ask question
        prompt = questions[0]

        # Prepare inputs
        inputs = processor(images=raw_image, text=prompt, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Generate response
        start_gen = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                num_beams=5,
                max_length=256,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1,
            )
        gen_time = time.time() - start_gen

        # Decode response
        response = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

        print(f"   Question: {prompt}")
        print(f"   Response: {response}")
        print(f"   Generation time: {gen_time:.2f}s")

        results.append({
            'image': img_path.name,
            'category': true_cat,
            'question': prompt,
            'response': response,
            'time': gen_time
        })

    # Summary
    print("\n" + "="*70)
    print("Quick Test Summary")
    print("="*70)
    print(f"\nModel: InstructBLIP-Vicuna-7B (4-bit NF4 quantized)")
    print(f"Parameters: {param_count / 1e9:.2f}B")
    print(f"Images tested: {len(results)}")

    avg_time = sum(r['time'] for r in results) / len(results)
    print(f"Average generation time: {avg_time:.2f}s")

    if torch.cuda.is_available():
        final_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"GPU Memory used: {final_memory:.2f} GB")

    print("\n" + "="*70)
    print("Quantization Comparison")
    print("="*70)
    print(f"\nInstructBLIP-7B:")
    print(f"  FP16 (full):    ~14GB memory (estimated)")
    print(f"  8-bit:          >8GB memory (failed)")
    print(f"  4-bit NF4:      ~{final_memory if torch.cuda.is_available() else 'N/A':.1f}GB memory")
    print(f"  Memory savings: ~{((14 - final_memory) / 14 * 100) if torch.cuda.is_available() else 'N/A':.0f}% vs FP16")

    print("\n" + "="*70)
    print("Comparison with Other Models")
    print("="*70)
    print(f"\nInstructBLIP-7B-4b: {param_count / 1e9:.2f}B params, ~{final_memory if torch.cuda.is_available() else 'N/A':.1f}GB memory")
    print(f"BLIP-2:             3.4B params, ~4.2GB memory")
    print(f"PaliGemma-8bit:     2.92B params, ~3.7GB memory")

    print("\nNote: 4-bit quantization may affect response quality")
    print("Check if quality degradation is acceptable for the memory savings")

    # Save results
    import json
    with open("instructblip_4bit_quick_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: instructblip_4bit_quick_test_results.json")

except Exception as e:
    print(f"\n[ERROR] {type(e).__name__}: {e}")
    print("\nPossible issues:")
    print("  - bitsandbytes not installed or outdated")
    print("  - Still insufficient GPU memory even with 4-bit")
    print("  - Model doesn't support 4-bit quantization")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)