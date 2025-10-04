"""
Test CheXagent-8b individually
Previous issue: Only outputs "What does it show?" (0% accuracy)
"""

import sys
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel, AutoModelForCausalLM

sys.path.append(str(Path(__file__).parent.parent.parent))

def test_chexagent():
    print("="*70)
    print("Testing: CheXagent-8b")
    print("="*70)

    model_id = "StanfordAIMI/CheXagent-8b"

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
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=True
        )
        print(f"[OK] Model loaded: {model.get_memory_footprint() / 1e9:.2f} GB")
    except Exception as e:
        print(f"[FAIL] Model loading failed: {e}")
        return False

    # Test with different questions and formats
    questions = [
        "What anatomical structures are visible in this ultrasound image?",
        "Describe this medical image.",
        "What type of scan is this?",
        "What do you see in this image?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\nTest {i}: {question}")

        # Format 1: Simple
        try:
            inputs = processor(images=image, text=question, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=100)
            answer = processor.decode(outputs[0], skip_special_tokens=True)
            print(f"  Answer: {answer}")

            # Check if it's just echoing the question
            if answer.strip() == question or "what does it show" in answer.lower():
                print(f"  [WARNING] Model echoing question or giving generic response")
                continue
            else:
                print(f"  [OK] Valid response")
                return True

        except Exception as e:
            print(f"  [FAIL] Error: {str(e)[:150]}")

    print("\n[FAIL] All attempts failed or only generic responses")
    return False

if __name__ == "__main__":
    success = test_chexagent()
    print("\n" + "="*70)
    print(f"Result: {'[OK] WORKING' if success else '[FAIL] STILL BROKEN'}")
    print("="*70)
