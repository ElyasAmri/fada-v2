"""
Test Qwen2.5-VL-3B individually
Previous error: AssertionError on inference
"""

import sys
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.append(str(Path(__file__).parent.parent.parent))

def test_qwen25vl_3b():
    print("="*70)
    print("Testing: Qwen2.5-VL-3B")
    print("="*70)

    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

    # Load test image
    test_image_path = Path("data/Fetal Ultrasound/Femur/Femur_001.png")
    if not test_image_path.exists():
        print(f"Error: Test image not found")
        return False

    image = Image.open(test_image_path).convert('RGB')
    print(f"Image loaded: {image.size}")

    # Load processor
    print("\nLoading processor...")
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        print("[OK] Processor loaded")
    except Exception as e:
        print(f"[FAIL] Processor loading failed: {e}")
        return False

    # Load model
    print("\nLoading model (4-bit)...")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=True
        )
        print(f"[OK] Model loaded: {model.get_memory_footprint() / 1e9:.2f} GB")
    except Exception as e:
        print(f"[FAIL] Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test inference
    question = "What anatomical structures are visible in this ultrasound image?"
    print(f"\nQuestion: {question}")

    # Try different formats
    print("\nAttempt 1: Simple format...")
    try:
        inputs = processor(images=image, text=question, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)
        answer = processor.decode(outputs[0], skip_special_tokens=True)
        print(f"[OK] Answer: {answer[:150]}...")
        return True
    except Exception as e:
        print(f"[FAIL] Simple format failed: {str(e)[:200]}")

    print("\nAttempt 2: Chat template...")
    try:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]
        }]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)
        answer = processor.decode(outputs[0], skip_special_tokens=True)
        print(f"[OK] Answer: {answer[:150]}...")
        return True
    except Exception as e:
        print(f"[FAIL] Chat template failed: {str(e)[:200]}")

    print("\nAttempt 3: <image> token...")
    try:
        prompt = f"<image>{question}"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)
        answer = processor.decode(outputs[0], skip_special_tokens=True)
        print(f"[OK] Answer: {answer[:150]}...")
        return True
    except Exception as e:
        print(f"[FAIL] <image> token failed: {str(e)[:200]}")

    print("\n[FAIL] All formats failed")
    return False

if __name__ == "__main__":
    success = test_qwen25vl_3b()
    print("\n" + "="*70)
    print(f"Result: {'[OK] WORKING' if success else '[FAIL] STILL BROKEN'}")
    print("="*70)
