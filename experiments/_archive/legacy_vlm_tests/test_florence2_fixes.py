"""Test Florence-2 with various fixes for dtype issues"""

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
from pathlib import Path
import time

print("="*70)
print("Florence-2 Fixes Testing")
print("="*70)

DATA_DIR = "data/Fetal Ultrasound"

# Get test image
test_images = []
for cat_dir in list(Path(DATA_DIR).iterdir())[:3]:
    if cat_dir.is_dir():
        images = list(cat_dir.glob("*.png"))[:1]
        if images:
            test_images.append(images[0])

def test_model_config(model_id, dtype_config, fix_name):
    """Test a specific configuration"""
    print(f"\n{'='*50}")
    print(f"Testing: {fix_name}")
    print(f"Model: {model_id}")
    print(f"Config: {dtype_config}")
    print("="*50)

    try:
        # Load model with specific config
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        if dtype_config == "float32":
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map="auto",
                attn_implementation="eager"
            )
        elif dtype_config == "float16":
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
                attn_implementation="eager"
            )
        elif dtype_config == "float16_forced":
            # Load in float16 and force everything
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
                attn_implementation="eager"
            )
            model = model.half()  # Force all weights to float16

        param_count = sum(p.numel() for p in model.parameters())
        print(f"[OK] Model loaded: {param_count / 1e9:.2f}B params")

        if torch.cuda.is_available():
            memory = torch.cuda.max_memory_allocated() / 1e9
            print(f"  GPU Memory: {memory:.2f} GB")

        # Test on sample image
        img_path = test_images[0]
        print(f"\nTesting on: {img_path.name}")
        image = Image.open(img_path)

        # Try captioning
        prompt = "<CAPTION>"
        inputs = processor(text=prompt, images=image, return_tensors="pt")

        # Get device
        device = next(model.parameters()).device

        # Apply different input processing based on fix
        if "forced" in dtype_config:
            # Force all tensors to half precision
            inputs = {
                k: v.to(device).half() if torch.is_tensor(v) and v.dtype == torch.float32
                else v.to(device) if torch.is_tensor(v)
                else v
                for k, v in inputs.items()
            }
        else:
            # Standard processing
            inputs = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in inputs.items()
            }

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

        print(f"[OK] SUCCESS! Generated caption:")
        print(f"  Caption: {caption}")
        print(f"  Time: {gen_time:.2f}s")

        # Test VQA if caption works
        print("\nTesting VQA...")
        vqa_prompt = "<VQA>What anatomical structures can you see in this ultrasound image?"
        inputs = processor(text=vqa_prompt, images=image, return_tensors="pt")

        if "forced" in dtype_config:
            inputs = {
                k: v.to(device).half() if torch.is_tensor(v) and v.dtype == torch.float32
                else v.to(device) if torch.is_tensor(v)
                else v
                for k, v in inputs.items()
            }
        else:
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=50,
                num_beams=3
            )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text,
            task="<VQA>",
            image_size=(image.width, image.height)
        )

        vqa_answer = parsed_answer.get("<VQA>", str(parsed_answer))
        print(f"  VQA Answer: {vqa_answer}")

        # Clean up
        del model
        torch.cuda.empty_cache()

        return True, caption

    except Exception as e:
        print(f"[FAIL] FAILED: {str(e)[:100]}")
        torch.cuda.empty_cache()
        return False, None

# Test all fixes
results = []

print("\n" + "="*70)
print("FIX 1: Force inputs to float16 with model.half()")
print("="*70)
success, caption = test_model_config(
    "microsoft/Florence-2-large",
    "float16_forced",
    "Fix 1: Force float16"
)
results.append(("Fix 1: Force float16", success))

print("\n" + "="*70)
print("FIX 2: Load model in float32 (no mixed precision)")
print("="*70)
success, caption = test_model_config(
    "microsoft/Florence-2-large",
    "float32",
    "Fix 2: Float32 model"
)
results.append(("Fix 2: Float32", success))

print("\n" + "="*70)
print("FIX 3: Use Florence-2-base (smaller model)")
print("="*70)
success, caption = test_model_config(
    "microsoft/Florence-2-base",
    "float16",
    "Fix 3: Base model"
)
results.append(("Fix 3: Base model", success))

print("\n" + "="*70)
print("FIX 4: Florence-2-base with float32")
print("="*70)
success, caption = test_model_config(
    "microsoft/Florence-2-base",
    "float32",
    "Fix 4: Base + float32"
)
results.append(("Fix 4: Base + float32", success))

# Summary
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
for fix_name, success in results:
    status = "[OK] SUCCESS" if success else "[FAIL] FAILED"
    print(f"{fix_name}: {status}")

successful = [r for r in results if r[1]]
if successful:
    print(f"\nSUCCESS: {len(successful)} fix(es) worked!")
    print("Florence-2 can be used with the right configuration")
else:
    print("\nFAILED: All fixes failed - Florence-2 has compatibility issues with current setup")