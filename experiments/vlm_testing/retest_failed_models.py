"""
Retest all previously failed VLM models with updated approaches
Following the MedGemma success, trying different loading methods and formats
"""

import sys
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel, AutoModelForCausalLM, AutoTokenizer
import traceback

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

def load_test_image():
    """Load a sample fetal ultrasound image"""
    test_image_path = Path("data/Fetal Ultrasound/Femur/Femur_001.png")
    if not test_image_path.exists():
        print(f"Error: Test image not found at {test_image_path}")
        return None
    return Image.open(test_image_path).convert('RGB')

def test_model(model_id, model_name, test_question="What anatomical structures are visible in this image?"):
    """Generic test function for VLM models"""
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"Model ID: {model_id}")
    print('='*70)

    image = load_test_image()
    if image is None:
        return False

    try:
        # Try loading with AutoModel + trust_remote_code (like MedGemma)
        print("\nAttempt 1: AutoModel with trust_remote_code...")
        try:
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModel.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                load_in_4bit=True
            )
            print(f"[OK] Loaded with AutoModel")
            print(f"   Memory: {model.get_memory_footprint() / 1e9:.2f} GB")

            # Try inference
            return try_inference(model, processor, image, test_question, model_name)

        except Exception as e1:
            print(f"[FAIL] AutoModel failed: {str(e1)[:100]}")

            # Try AutoModelForCausalLM (like MedGemma)
            print("\nAttempt 2: AutoModelForCausalLM...")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    load_in_4bit=True
                )
                print(f"[OK] Loaded with AutoModelForCausalLM")
                print(f"   Memory: {model.get_memory_footprint() / 1e9:.2f} GB")

                return try_inference(model, processor, image, test_question, model_name)

            except Exception as e2:
                print(f"[FAIL] AutoModelForCausalLM failed: {str(e2)[:100]}")
                print(f"\n[FAIL] Model {model_name} cannot be loaded")
                return False

    except Exception as e:
        print(f"[FAIL] Fatal error: {str(e)[:100]}")
        traceback.print_exc()
        return False

def try_inference(model, processor, image, question, model_name):
    """Try different inference formats"""

    # Format 1: Simple text + image
    print("\n  Testing inference format 1: Simple text...")
    try:
        inputs = processor(images=image, text=question, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)
        answer = processor.decode(outputs[0], skip_special_tokens=True)
        print(f"  [OK] Answer: {answer[:100]}...")
        return True
    except Exception as e:
        print(f"  [FAIL] Format 1 failed: {str(e)[:80]}")

    # Format 2: Chat template (like MedGemma)
    print("\n  Testing inference format 2: Chat template...")
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
        print(f"  [OK] Answer: {answer[:100]}...")
        return True
    except Exception as e:
        print(f"  [FAIL] Format 2 failed: {str(e)[:80]}")

    # Format 3: <image> token
    print("\n  Testing inference format 3: <image> token...")
    try:
        prompt = f"<image>{question}"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)
        answer = processor.decode(outputs[0], skip_special_tokens=True)
        print(f"  [OK] Answer: {answer[:100]}...")
        return True
    except Exception as e:
        print(f"  [FAIL] Format 3 failed: {str(e)[:80]}")

    print(f"\n  [FAIL] All inference formats failed for {model_name}")
    return False

def main():
    """Retest all failed models"""

    # Test first batch first - most promising models
    failed_models = [
        ("Qwen/Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-3B"),
        ("deepseek-ai/deepseek-vl-1.3b-chat", "DeepSeek-VL-1.3B"),
        ("StanfordAIMI/CheXagent-8b", "CheXagent-8b"),
    ]

    # Batch 2 (if time permits):
    # ("Qwen/Qwen-VL-Chat", "Qwen-VL-Chat"),
    # ("MAGAer13/mplug-owl2-llama2-7b", "mPLUG-Owl2"),
    # ("Vision-CAIR/TinyGPT-V", "TinyGPT-V"),
    # ("THUDM/cogvlm-chat-hf", "CogVLM-chat-hf"),
    # ("THUDM/cogagent-chat-hf", "CogAgent-chat-hf"),
    # ("adept/fuyu-8b", "Fuyu-8B"),

    results = {}

    for model_id, model_name in failed_models:
        success = test_model(model_id, model_name)
        results[model_name] = "[OK] Works" if success else "[FAIL] Still Failed"

    # Summary
    print(f"\n\n{'='*70}")
    print("RETEST SUMMARY")
    print('='*70)
    for model_name, status in results.items():
        print(f"{model_name:30} {status}")

    working_count = sum(1 for status in results.values() if "[OK]" in status)
    print(f"\n{working_count}/{len(failed_models)} models now working")
    print('='*70)

if __name__ == "__main__":
    main()
