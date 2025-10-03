"""Test Florence-2 with transformers 4.36.2 in new environment"""

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
from pathlib import Path
import time
import sys

print("="*70)
print("Florence-2 Test with Transformers 4.36.2")
print("="*70)

# Check versions
import transformers
print(f"\nEnvironment check:")
print(f"  Python: {sys.version}")
print(f"  PyTorch: {torch.__version__}")
print(f"  Transformers: {transformers.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")

DATA_DIR = "data/Fetal Ultrasound"
MODEL_ID = "microsoft/Florence-2-large"

try:
    print(f"\n1. Loading Florence-2-large...")
    print(f"   Model: {MODEL_ID}")

    # Load processor and model with attention implementation workaround
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Try loading without flash attention
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        attn_implementation="eager"  # Use eager attention instead of flash
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"   [OK] Model loaded: {param_count / 1e9:.2f}B params")

    if torch.cuda.is_available():
        memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"   GPU Memory: {memory:.2f} GB")

    print("\n2. Testing on sample images...")

    # Get test images
    test_images = []
    for cat_dir in list(Path(DATA_DIR).iterdir())[:5]:
        if cat_dir.is_dir():
            images = list(cat_dir.glob("*.png"))[:1]
            if images:
                test_images.append(images[0])

    for i, img_path in enumerate(test_images[:3]):
        print(f"\n   Image {i+1}: {img_path.name}")
        print(f"   Category: {img_path.parent.name}")

        image = Image.open(img_path).convert('RGB')

        # Test caption task
        prompt = "<CAPTION>"
        inputs = processor(text=prompt, images=image, return_tensors="pt")

        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        # Generate
        start = time.time()
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=50,
                num_beams=3
            )
        gen_time = time.time() - start

        # Decode
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        # Parse output
        parsed_answer = processor.post_process_generation(
            generated_text,
            task=prompt,
            image_size=(image.width, image.height)
        )

        caption = parsed_answer.get("<CAPTION>", str(parsed_answer))
        print(f"   Caption: {caption}")
        print(f"   Time: {gen_time:.2f}s")

    print("\n3. Testing VQA...")

    # Test VQA on one image
    img_path = test_images[0]
    image = Image.open(img_path).convert('RGB')

    vqa_prompt = "<VQA>What anatomical structures can you see in this ultrasound image?"
    inputs = processor(text=vqa_prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

    start = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=100,
            num_beams=3
        )
    gen_time = time.time() - start

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task="<VQA>",
        image_size=(image.width, image.height)
    )

    vqa_answer = parsed_answer.get("<VQA>", str(parsed_answer))
    print(f"\n   VQA Question: What anatomical structures can you see?")
    print(f"   VQA Answer: {vqa_answer}")
    print(f"   Time: {gen_time:.2f}s")

    # Test detailed caption
    print("\n4. Testing DETAILED_CAPTION...")

    prompt = "<DETAILED_CAPTION>"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=150,
            num_beams=3
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=prompt,
        image_size=(image.width, image.height)
    )

    detailed_caption = parsed_answer.get("<DETAILED_CAPTION>", str(parsed_answer))
    print(f"   Detailed Caption: {detailed_caption}")

    print("\n" + "="*70)
    print("SUCCESS! Florence-2 works with transformers 4.36.2")
    print("="*70)

    print(f"\nModel Summary:")
    print(f"  - Florence-2-large: {param_count / 1e9:.2f}B params")
    print(f"  - Memory usage: {memory:.2f} GB")
    print(f"  - Supports: Captioning, VQA, Detailed Captioning")
    print(f"  - Extremely efficient for its capabilities")

except Exception as e:
    print(f"\n[ERROR] {type(e).__name__}: {e}")
    print("\nDebugging info:")
    import traceback
    traceback.print_exc()

    print("\nPossible solutions:")
    print("1. Ensure transformers==4.36.2 is installed")
    print("2. Try with Florence-2-base instead")
    print("3. Check CUDA compatibility")

print("\n" + "="*70)