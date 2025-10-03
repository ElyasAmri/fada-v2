"""Quick test of Moondream2 on FADA dataset"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from pathlib import Path
import time

print("="*70)
print("Moondream2 Quick Test on FADA Dataset")
print("="*70)

# Configuration
MODEL_ID = "vikhyatk/moondream2"
MODEL_REVISION = "2025-06-21"
DATA_DIR = "data/Fetal Ultrasound"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n1. Loading Moondream2 model...")
print(f"   Model: {MODEL_ID}")
print(f"   Revision: {MODEL_REVISION}")
print(f"   Device: {device}")

start = time.time()

try:
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        device_map={"": device}
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

    # Collect test images (3 per category, first 5 categories)
    test_images = []
    category_dirs = [d for d in Path(DATA_DIR).iterdir() if d.is_dir()]

    for cat_dir in category_dirs[:5]:
        cat_name = cat_dir.name
        images = list(cat_dir.glob("*.png")) + list(cat_dir.glob("*.jpg"))

        if not images:
            continue

        for img_path in images[:3]:
            test_images.append({
                'path': img_path,
                'category': cat_name
            })

    print(f"   Testing {len(test_images)} images...")

    # Test questions for ultrasound images
    questions = [
        "What anatomical structures can you see in this ultrasound image?",
        "Describe this fetal ultrasound image.",
        "What part of the fetus is shown in this image?"
    ]

    results = []

    for i, item in enumerate(test_images[:5]):  # Test first 5 for quick test
        img_path = item['path']
        true_cat = item['category']

        print(f"\n   Image {i+1}/{5}: {img_path.name}")
        print(f"   True category: {true_cat}")

        # Load image
        image = Image.open(img_path)

        # Ask question
        question = questions[0]  # Use first question

        # Generate response
        start_gen = time.time()
        result = model.query(image, question)
        response = result["answer"]
        gen_time = time.time() - start_gen

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
    print(f"\nModel: Moondream2 ({MODEL_REVISION})")
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
    print(f"\nMoondream2:       {param_count / 1e9:.2f}B params, ~{final_memory if torch.cuda.is_available() else 'N/A':.1f}GB memory")
    print(f"SmolVLM-500M:     0.51B params, ~1.0GB memory")
    print(f"BLIP-2:           3.4B params, ~4.2GB memory")

    print("\nNote: Moondream2 is optimized for edge deployment")
    print("Check response quality for medical VQA suitability")

    # Save results
    import json
    with open("moondream_quick_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: moondream_quick_test_results.json")

except Exception as e:
    print(f"\n[ERROR] {type(e).__name__}: {e}")
    print("\nPossible issues:")
    print("  - Model not found or access denied")
    print("  - Insufficient GPU memory")
    print("  - Incompatible transformers version")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
