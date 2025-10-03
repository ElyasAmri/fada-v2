"""
Quick test script to verify VLM interface works before running full web app
"""

import sys
from pathlib import Path
from PIL import Image

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.inference.vlm_interface import create_top_vlms
from src.data.question_loader import get_question_loader

def test_vlm_interface():
    """Test VLM interface with a sample image"""

    # Load sample image
    test_image_path = Path("data/Fetal Ultrasound/Femur/Femur_001.png")
    if not test_image_path.exists():
        print(f"Error: Test image not found at {test_image_path}")
        return False

    print(f"Loading test image: {test_image_path}")
    image = Image.open(test_image_path).convert('RGB')
    print(f"Image size: {image.size}")

    # Load questions
    print("\nLoading questions...")
    question_loader = get_question_loader()
    questions = question_loader.get_questions()
    print(f"Loaded {len(questions)} questions")

    # Use only first 2 questions for quick test
    test_questions = questions[:2]
    print(f"\nTest questions:")
    for i, q in enumerate(test_questions):
        print(f"  Q{i+1}: {q[:60]}...")

    # Create VLM manager
    print("\nInitializing VLM manager (8GB GPU mode)...")
    manager = create_top_vlms(use_api=False, gpu_8gb=True)

    # Test each model
    model_keys = ["minicpm", "internvl2_2b", "moondream"]

    for model_key in model_keys:
        print(f"\n{'='*60}")
        print(f"Testing: {model_key}")
        print('='*60)

        model = manager.get_model(model_key)
        if model is None:
            print(f"Error: Model {model_key} not found in manager")
            continue

        print(f"Model name: {model.model_name}")
        print(f"Is loaded: {model.is_loaded}")

        # Load model
        print("Loading model...")
        try:
            model.load()
            print(f"Model loaded successfully")
            print(f"Is loaded: {model.is_loaded}")
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Test inference with first question
        print(f"\nTesting inference with question: {test_questions[0][:40]}...")
        try:
            answer = model.answer_question(image, test_questions[0])
            print(f"Answer: {answer[:100]}...")
        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()

        # Unload model
        print("\nUnloading model...")
        try:
            model.unload()
            print(f"Model unloaded successfully")
            print(f"Is loaded: {model.is_loaded}")
        except Exception as e:
            print(f"Error unloading model: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("Test complete!")
    print('='*60)

if __name__ == "__main__":
    test_vlm_interface()
