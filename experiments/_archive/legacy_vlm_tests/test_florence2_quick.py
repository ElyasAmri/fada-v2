"""Test Florence-2 on FADA dataset for VQA"""

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
from pathlib import Path
import time

print("="*70)
print("Florence-2 Quick Test on FADA Dataset")
print("="*70)

# Configuration - try large version first (0.7B), can fall back to base (0.2B)
MODEL_ID = "microsoft/Florence-2-large"  # or "microsoft/Florence-2-base" for smaller
DATA_DIR = "data/Fetal Ultrasound"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n1. Loading Florence-2 model...")
print(f"   Model: {MODEL_ID}")
print(f"   Device: {device}")

start = time.time()

try:
    # Load processor and model - use specific Florence2 classes
    from transformers import AutoProcessor, AutoModelForCausalLM

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Try loading with attn_implementation set
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        attn_implementation="eager"  # Use eager attention to avoid SDPA issue
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

    # Florence-2 uses task prompts
    # Try different task prompts to see which works best
    task_prompts = [
        "<VQA>",  # Visual Question Answering task
        "<DETAILED_CAPTION>",  # Detailed captioning
        "<MORE_DETAILED_CAPTION>",  # More detailed captioning
        "<CAPTION>",  # Basic caption
    ]

    results = []

    # Test VQA task first
    vqa_question = "What anatomical structures can you see in this ultrasound image?"

    for i, item in enumerate(test_images[:5]):
        img_path = item['path']
        true_cat = item['category']

        print(f"\n   Image {i+1}/{5}: {img_path.name}")
        print(f"   True category: {true_cat}")

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Try VQA task
        prompt = "<VQA>"
        text_input = vqa_question

        # Process inputs
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )

        # Add the question for VQA task
        if prompt == "<VQA>":
            # For VQA, we need to provide the question
            inputs = processor(
                text=f"{prompt}{vqa_question}",
                images=image,
                return_tensors="pt"
            )

        if torch.cuda.is_available():
            # Move to cuda with correct dtype
            inputs = {k: v.to("cuda").to(torch.float16) if v.dtype == torch.float32 else v.to("cuda")
                     for k, v in inputs.items()}

        # Generate response
        start_gen = time.time()
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=100,
                num_beams=3,
                do_sample=False
            )
        gen_time = time.time() - start_gen

        # Decode response
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        # Parse the output - Florence-2 returns structured outputs
        parsed_answer = processor.post_process_generation(
            generated_text,
            task=prompt,
            image_size=(image.width, image.height)
        )

        # Extract response based on task
        if prompt == "<VQA>" and isinstance(parsed_answer, dict):
            response = parsed_answer.get(prompt, str(parsed_answer))
        else:
            response = str(parsed_answer)

        print(f"   Task: {prompt}")
        print(f"   Question: {vqa_question if prompt == '<VQA>' else 'N/A'}")
        print(f"   Response: {response}")
        print(f"   Generation time: {gen_time:.2f}s")

        results.append({
            'image': img_path.name,
            'category': true_cat,
            'task': prompt,
            'question': vqa_question if prompt == '<VQA>' else None,
            'response': response,
            'time': gen_time
        })

    # Also test with detailed caption task for comparison
    print("\n3. Testing with DETAILED_CAPTION task for comparison...")

    for i, item in enumerate(test_images[:2]):  # Just 2 images for comparison
        img_path = item['path']
        image = Image.open(img_path).convert('RGB')

        prompt = "<DETAILED_CAPTION>"
        inputs = processor(text=prompt, images=image, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        start_gen = time.time()
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=100, num_beams=3)
        gen_time = time.time() - start_gen

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text,
            task=prompt,
            image_size=(image.width, image.height)
        )

        response = parsed_answer.get(prompt, str(parsed_answer))

        print(f"\n   Image: {img_path.name}")
        print(f"   Detailed Caption: {response}")

    # Summary
    print("\n" + "="*70)
    print("Quick Test Summary")
    print("="*70)
    print(f"\nModel: Florence-2")
    print(f"Parameters: {param_count / 1e9:.2f}B")
    print(f"Images tested: {len(results)}")

    avg_time = sum(r['time'] for r in results) / len(results) if results else 0
    print(f"Average generation time: {avg_time:.2f}s")

    if torch.cuda.is_available():
        final_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"GPU Memory used: {final_memory:.2f} GB")

    print("\n" + "="*70)
    print("Comparison with Other Models")
    print("="*70)
    print(f"\nFlorence-2:        {param_count / 1e9:.2f}B params, ~{final_memory if torch.cuda.is_available() else 'N/A':.1f}GB memory")
    print(f"LLaVA-NeXT-7B-4b:  3.92B params, ~7.6GB memory")
    print(f"BLIP-2:            3.4B params, ~4.2GB memory")
    print(f"SmolVLM-256M:      0.26B params, ~1.0GB memory")

    print("\nNote: Florence-2 is a unified model for multiple vision tasks")
    print("Extremely efficient with strong zero-shot performance")

    # Save results
    import json
    with open("florence2_quick_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: florence2_quick_test_results.json")

except Exception as e:
    print(f"\n[ERROR] {type(e).__name__}: {e}")
    print("\nPossible issues:")
    print("  - Model requires trust_remote_code=True")
    print("  - May need specific transformers version")
    print("  - Florence-2 uses special task prompts")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)