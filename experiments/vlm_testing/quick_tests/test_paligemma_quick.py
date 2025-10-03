"""Quick test of PaliGemma-3B-mix on FADA dataset"""

import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
from pathlib import Path
import time

print("="*70)
print("PaliGemma-3B-mix Quick Test on FADA Dataset")
print("="*70)

# Configuration
MODEL_ID = "google/paligemma-3b-mix-224"
DATA_DIR = "data/Fetal Ultrasound"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n1. Loading PaliGemma-3B-mix model...")
print(f"   Model: {MODEL_ID}")
print(f"   Device: {device}")

start = time.time()

try:
    # Load processor and model
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = PaliGemmaForConditionalGeneration.from_pretrained(MODEL_ID).eval()

    if torch.cuda.is_available():
        model = model.to("cuda")

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

        # Load image and convert to RGB (PaliGemma requires 3-channel images)
        image = Image.open(img_path).convert('RGB')

        # Ask question
        prompt = questions[0]

        # Prepare inputs
        model_inputs = processor(text=prompt, images=image, return_tensors="pt")

        if torch.cuda.is_available():
            model_inputs = {k: v.to("cuda") for k, v in model_inputs.items()}

        input_len = model_inputs["input_ids"].shape[-1]

        # Generate response
        start_gen = time.time()
        with torch.inference_mode():
            generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        gen_time = time.time() - start_gen

        # Decode response
        response = processor.decode(generation[0][input_len:], skip_special_tokens=True)

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
    print(f"\nModel: PaliGemma-3B-mix-224")
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
    print(f"SmolVLM-256M:     0.26B params, ~1.0GB memory")
    print(f"BLIP-VQA-base:    0.36B params, ~1.5GB memory")
    print(f"SmolVLM-500M:     0.51B params, ~1.0GB memory")
    print(f"Moondream2:       1.93B params, ~4.5GB memory")
    print(f"PaliGemma-3B:     {param_count / 1e9:.2f}B params, ~{final_memory if torch.cuda.is_available() else 'N/A':.1f}GB memory")
    print(f"BLIP-2:           3.4B params, ~4.2GB memory")

    print("\nNote: PaliGemma is Google's lightweight VLM (SigLIP + Gemma)")
    print("Multi-task fine-tuned model (not conversational)")

    # Save results
    import json
    with open("paligemma_quick_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: paligemma_quick_test_results.json")

except Exception as e:
    print(f"\n[ERROR] {type(e).__name__}: {e}")
    print("\nPossible issues:")
    print("  - Model not found or access denied")
    print("  - Insufficient GPU memory (~6-8GB needed)")
    print("  - Incompatible transformers version")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
