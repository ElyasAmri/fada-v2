"""Test Microsoft Kosmos-2 vision-language model"""

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
from pathlib import Path
import time

print("="*70)
print("Kosmos-2 Test")
print("="*70)

# Check CUDA
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

DATA_DIR = "data/Fetal Ultrasound"
MODEL_ID = "microsoft/kosmos-2-patch14-224"

try:
    print(f"\n1. Loading Kosmos-2...")
    print(f"   Model: {MODEL_ID}")
    start_time = time.time()

    # Load processor and model
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    load_time = time.time() - start_time
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   [OK] Model loaded: {param_count / 1e9:.2f}B params")
    print(f"   Load time: {load_time:.1f}s")

    if torch.cuda.is_available():
        memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"   GPU Memory: {memory:.2f} GB")

    print("\n2. Testing on fetal ultrasound images...")

    # Get test images from different categories
    test_images = []
    categories = ["Abodomen", "Aorta", "Brain", "Femur", "Heart"]
    for cat in categories:
        cat_dir = Path(DATA_DIR) / cat
        if cat_dir.exists():
            images = list(cat_dir.glob("*.png"))[:2]  # 2 per category
            test_images.extend(images)

    if not test_images:
        print("   No test images found!")
    else:
        print(f"   Testing {len(test_images)} images...")

        correct_anatomy = 0
        fetal_context = 0

        for i, img_path in enumerate(test_images[:10]):  # Test up to 10
            print(f"\n   Image {i+1}/{min(len(test_images), 10)}: {img_path.name}")
            print(f"   Category: {img_path.parent.name}")

            image = Image.open(img_path).convert('RGB')

            # Test with medical VQA prompt
            prompt = "<grounding>What anatomical structures can you see in this ultrasound image?"

            inputs = processor(text=prompt, images=image, return_tensors="pt")

            # Move to device
            device = next(model.parameters()).device
            inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

            # Generate
            start = time.time()
            with torch.no_grad():
                generated_ids = model.generate(
                    pixel_values=inputs["pixel_values"],
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    image_embeds=None,
                    image_embeds_position_mask=inputs["image_embeds_position_mask"],
                    use_cache=True,
                    max_new_tokens=128,
                )
            gen_time = time.time() - start

            # Decode
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Clean response
            processed_text = processor.post_process_generation(generated_text, cleanup_and_extract=False)

            print(f"   Response: {processed_text}")
            print(f"   Time: {gen_time:.2f}s")

            # Check for medical understanding
            response_lower = processed_text.lower()
            if any(term in response_lower for term in ['fetal', 'fetus', 'baby', 'pregnancy', 'ultrasound']):
                fetal_context += 1

            category_lower = img_path.parent.name.lower()
            if category_lower in response_lower or \
               (category_lower == "abodomen" and "abdomen" in response_lower):
                correct_anatomy += 1

        # Calculate accuracy
        print(f"\n3. Accuracy Assessment:")
        print(f"   Fetal context recognition: {fetal_context}/{min(len(test_images), 10)} ({fetal_context/min(len(test_images), 10)*100:.1f}%)")
        print(f"   Anatomy identification: {correct_anatomy}/{min(len(test_images), 10)} ({correct_anatomy/min(len(test_images), 10)*100:.1f}%)")

    # Test different Kosmos-2 capabilities
    print("\n4. Testing Kosmos-2 Special Features...")

    if test_images:
        img_path = test_images[0]
        image = Image.open(img_path).convert('RGB')

        # Test different prompt types
        test_prompts = [
            "<grounding>Describe this medical image",
            "<grounding>Is this a normal or abnormal ultrasound?",
            "<grounding>What type of medical scan is this?",
            "<grounding>Identify the fetal structures",
        ]

        for prompt in test_prompts:
            print(f"\n   Prompt: {prompt}")

            inputs = processor(text=prompt, images=image, return_tensors="pt")
            inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = model.generate(
                    pixel_values=inputs["pixel_values"],
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    image_embeds=None,
                    image_embeds_position_mask=inputs["image_embeds_position_mask"],
                    use_cache=True,
                    max_new_tokens=64,
                )

            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            processed_text = processor.post_process_generation(generated_text, cleanup_and_extract=False)

            print(f"   Response: {processed_text[:200]}")

    print("\n" + "="*70)
    print("Kosmos-2 Test Complete")
    print("="*70)

    print(f"\nModel Summary:")
    print(f"  - Kosmos-2: {param_count / 1e9:.2f}B params")
    print(f"  - Memory: {memory:.2f} GB")
    print(f"  - Supports: Grounding, VQA, Image-Text Understanding")
    print(f"  - Fetal context: {'YES' if fetal_context > len(test_images)*0.3 else 'LIMITED'}")
    print(f"  - Medical accuracy potential: {'HIGH' if correct_anatomy > len(test_images)*0.5 else 'MODERATE'}")

except Exception as e:
    print(f"\n[ERROR] {type(e).__name__}: {e}")
    print("\nTrying alternative Kosmos-2 model...")

    # Try original Kosmos-2
    try:
        MODEL_ID = "microsoft/kosmos-2"
        print(f"\nLoading {MODEL_ID}...")

        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16
        )

        print("Alternative Kosmos-2 loaded successfully!")

    except Exception as e2:
        print(f"Also failed: {e2}")

print("\n" + "="*70)