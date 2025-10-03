"""Quick test of SmolVLM-256M on FADA dataset - world's smallest VLM"""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
from pathlib import Path
import time

print("="*70)
print("SmolVLM-256M Quick Test on FADA Dataset")
print("="*70)

# Configuration
MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
DATA_DIR = "data/Fetal Ultrasound"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n1. Loading SmolVLM-256M model (world's smallest VLM)...")
print(f"   Model: {MODEL_ID}")
print(f"   Device: {device}")

start = time.time()

try:
    # Load processor and model
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
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

        question = questions[0]

        # Prepare messages with image path
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": str(img_path)},
                    {"type": "text", "text": question}
                ]
            }
        ]

        # Process inputs
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate response
        start_gen = time.time()
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=100
            )
        gen_time = time.time() - start_gen

        # Decode response
        generated_texts = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )

        response = generated_texts[0]
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        elif question in response:
            response = response.split(question)[-1].strip()

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
    print(f"\nModel: SmolVLM-256M-Instruct (World's Smallest VLM)")
    print(f"Parameters: {param_count / 1e9:.2f}B")
    print(f"Images tested: {len(results)}")

    avg_time = sum(r['time'] for r in results) / len(results)
    print(f"Average generation time: {avg_time:.2f}s")

    if torch.cuda.is_available():
        final_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"GPU Memory used: {final_memory:.2f} GB")

    print("\n" + "="*70)
    print("Comparison with Other Tested Models")
    print("="*70)
    print(f"\nVILT-b32:         0.12B params, ~0.5GB memory")
    print(f"SmolVLM-256M:     {param_count / 1e9:.2f}B params, ~{final_memory if torch.cuda.is_available() else 'N/A':.1f}GB memory")
    print(f"BLIP-VQA-base:    0.36B params, ~1.5GB memory")
    print(f"SmolVLM-500M:     0.51B params, ~1.0GB memory")
    print(f"Moondream2:       1.93B params, ~4.5GB memory")
    print(f"BLIP-2:           3.4B params, ~4.2GB memory")

    print("\nNote: SmolVLM-256M is HuggingFace's smallest VLM (<1GB RAM)")
    print("Check if extreme efficiency compensates for smaller capacity")

    # Save results
    import json
    with open("smolvlm256_quick_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: smolvlm256_quick_test_results.json")

except Exception as e:
    print(f"\n[ERROR] {type(e).__name__}: {e}")
    print("\nPossible issues:")
    print("  - Missing dependencies (num2words)")
    print("  - Incompatible transformers version")
    print("  - Insufficient GPU memory")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
