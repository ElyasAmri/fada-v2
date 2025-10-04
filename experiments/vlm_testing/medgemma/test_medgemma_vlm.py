"""
Test MedGemma multimodal (vision-language) model
"""

import sys
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

# Add src to path
sys.path.append(str(Path(__file__).parent))

def test_medgemma_vlm():
    """Test MedGemma-4B multimodal model"""

    print("="*70)
    print("MedGemma-4B Multimodal VLM Test")
    print("="*70)

    # Model ID
    model_id = "google/medgemma-4b-it"

    # Load sample image
    test_image_path = Path("data/Fetal Ultrasound/Femur/Femur_001.png")
    if not test_image_path.exists():
        print(f"Error: Test image not found at {test_image_path}")
        return False

    print(f"\nLoading test image: {test_image_path}")
    image = Image.open(test_image_path).convert('RGB')
    print(f"Image size: {image.size}")

    # Load processor and model
    print(f"\nLoading MedGemma-4B multimodal model...")
    print("This may take a few minutes on first run...")

    try:
        # Load processor
        print("\n1. Loading processor...")
        processor = AutoProcessor.from_pretrained(model_id)
        print("   [OK] Processor loaded")

        # Check if processor has image_token
        if hasattr(processor, 'image_token'):
            print(f"   Image token: {processor.image_token}")
        if hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, 'special_tokens_map'):
            print(f"   Special tokens: {processor.tokenizer.special_tokens_map}")

        # Load model with 4-bit quantization
        print("\n2. Loading model (4-bit quantization)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=True
        )
        print("   [OK] Model loaded")
        print(f"   Memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")

        # Test inference
        print("\n3. Testing inference on fetal ultrasound...")

        test_questions = [
            "What anatomical structures are visible in this ultrasound image?",
            "Describe what you see in this medical image.",
            "Is this image showing a normal or abnormal finding?"
        ]

        for i, question in enumerate(test_questions, 1):
            print(f"\n   Question {i}: {question}")

            # Try different prompt formats
            # MedGemma might use special formatting
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                }
            ]

            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

            inputs = processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(model.device)

            # Generate answer
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False
                )

            # Decode answer
            answer = processor.decode(outputs[0], skip_special_tokens=True)

            # Remove the question from the answer if present
            if question in answer:
                answer = answer.replace(question, "").strip()

            print(f"   Answer: {answer[:200]}...")

        print("\n" + "="*70)
        print("[SUCCESS] MedGemma multimodal is working!")
        print("="*70)

        return True

    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        print("\nPossible issues:")
        print("  - Model access not granted (need to accept terms)")
        print("  - Insufficient GPU memory")
        print("  - Wrong model class (try PaliGemmaForConditionalGeneration)")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_medgemma_vlm()
