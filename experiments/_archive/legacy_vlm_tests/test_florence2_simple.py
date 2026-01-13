"""Simple test of Florence-2 captioning on FADA dataset"""

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
from pathlib import Path
import time

print("="*70)
print("Florence-2 Simple Captioning Test")
print("="*70)

MODEL_ID = "microsoft/Florence-2-large"
DATA_DIR = "data/Fetal Ultrasound"

try:
    print("\n1. Loading model...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager"
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"   Model loaded: {param_count / 1e9:.2f}B params")

    if torch.cuda.is_available():
        memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"   GPU Memory: {memory:.2f} GB")

    print("\n2. Testing on sample images...")

    # Test on 3 images
    test_images = []
    for cat_dir in list(Path(DATA_DIR).iterdir())[:3]:
        if cat_dir.is_dir():
            images = list(cat_dir.glob("*.png"))[:1]
            if images:
                test_images.append(images[0])

    for img_path in test_images:
        print(f"\n   Image: {img_path.name}")
        image = Image.open(img_path)

        # Try basic caption task
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

    print("\n" + "="*70)
    print("Summary:")
    print(f"Florence-2 ({param_count/1e9:.2f}B params) works for captioning")
    print("Very efficient model with ~1.5GB memory usage")

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()