"""Quick test of Qwen2-VL-7B-Instruct-GPTQ-Int8 on FADA dataset"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
from pathlib import Path
import time

print("="*70)
print("Qwen2-VL-7B-Instruct-GPTQ-Int8 Quick Test on FADA Dataset")
print("="*70)

# Configuration
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8"
DATA_DIR = "data/Fetal Ultrasound"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n1. Loading Qwen2-VL-7B model (8-bit quantized)...")
print(f"   Model: {MODEL_ID}")
print(f"   Device: {device}")
print(f"   Quantization: GPTQ Int8")

start = time.time()

try:
    # Load processor and model
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
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
        image = Image.open(img_path)

        # Ask question
        question = questions[0]

        # Prepare messages
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }]

        # Process inputs
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Generate response
        start_gen = time.time()
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
        gen_time = time.time() - start_gen

        # Decode response
        input_len = inputs["input_ids"].shape[1]
        generated_ids_trimmed = generated_ids[:, input_len:]
        response = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        print(f"   Question: {question}")
        print(f"   Response: {response}")
        print(f"   Generation time: {gen_time:.2f}s")

        results.append({
            'image': img_path.name,
            'category': true_cat,
            'question': question,
            'response': response,
            'time': gen_time
        })

    # Summary
    print("\n" + "="*70)
    print("Quick Test Summary")
    print("="*70)
    print(f"\nModel: Qwen2-VL-7B-Instruct-GPTQ-Int8")
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
    print(f"\nQwen2-VL-7B-Int8: {param_count / 1e9:.2f}B params, ~{final_memory if torch.cuda.is_available() else 'N/A':.1f}GB memory")
    print(f"BLIP-2:           3.4B params, ~4.2GB memory")
    print(f"PaliGemma-8bit:   2.92B params, ~3.7GB memory")
    print(f"Moondream2:       1.93B params, ~4.5GB memory")

    print("\nNote: Qwen2-VL is SOTA for visual understanding benchmarks")
    print("Check response quality for medical VQA suitability")

    # Save results
    import json
    with open("qwen2vl_quick_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: qwen2vl_quick_test_results.json")

except Exception as e:
    print(f"\n[ERROR] {type(e).__name__}: {e}")
    print("\nPossible issues:")
    print("  - Need transformers from source: pip install git+https://github.com/huggingface/transformers")
    print("  - Missing qwen-vl-utils: pip install qwen-vl-utils")
    print("  - Insufficient GPU memory")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
