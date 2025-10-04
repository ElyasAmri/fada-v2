"""
Test DeepSeek-VL-1.3B individually
Previous error: Model type 'multi_modality' not recognized
"""

import sys
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel, AutoModelForCausalLM

sys.path.append(str(Path(__file__).parent.parent.parent))

def test_deepseek_vl():
    print("="*70)
    print("Testing: DeepSeek-VL-1.3B")
    print("="*70)

    model_id = "deepseek-ai/deepseek-vl-1.3b-chat"

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

    # Try AutoModel with trust_remote_code
    print("\nAttempt 1: AutoModel with trust_remote_code...")
    try:
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=True
        )
        print(f"[OK] Model loaded: {model.get_memory_footprint() / 1e9:.2f} GB")

        # Test inference
        question = "What anatomical structures are visible in this ultrasound image?"
        print(f"\nQuestion: {question}")

        try:
            inputs = processor(images=image, text=question, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=100)
            answer = processor.decode(outputs[0], skip_special_tokens=True)
            print(f"[OK] Answer: {answer[:150]}...")
            return True
        except Exception as e:
            print(f"[FAIL] Inference failed: {str(e)[:200]}")
            return False

    except Exception as e:
        print(f"[FAIL] AutoModel failed: {str(e)[:200]}")

    # Try AutoModelForCausalLM
    print("\nAttempt 2: AutoModelForCausalLM...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=True
        )
        print(f"[OK] Model loaded: {model.get_memory_footprint() / 1e9:.2f} GB")

        # Test inference
        question = "What anatomical structures are visible in this ultrasound image?"
        print(f"\nQuestion: {question}")

        try:
            inputs = processor(images=image, text=question, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=100)
            answer = processor.decode(outputs[0], skip_special_tokens=True)
            print(f"[OK] Answer: {answer[:150]}...")
            return True
        except Exception as e:
            print(f"[FAIL] Inference failed: {str(e)[:200]}")
            return False

    except Exception as e:
        print(f"[FAIL] AutoModelForCausalLM failed: {str(e)[:200]}")

    return False

if __name__ == "__main__":
    success = test_deepseek_vl()
    print("\n" + "="*70)
    print(f"Result: {'[OK] WORKING' if success else '[FAIL] STILL BROKEN'}")
    print("="*70)
