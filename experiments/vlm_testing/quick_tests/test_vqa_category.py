"""Test VQA model for a specific category"""
import sys
from pathlib import Path
from PIL import Image
from src.models.vqa_model import UltrasoundVQA

def test_category_vqa(category_name, model_dir, image_dir, num_test=1):
    """Test VQA model on sample images from category"""

    print(f"\n{'='*60}")
    print(f"Testing VQA Model: {category_name}")
    print(f"{'='*60}\n")

    # Initialize model
    vqa = UltrasoundVQA(model_path=model_dir)
    vqa.load_model()

    # Get test images
    image_path = Path(image_dir)
    images = list(image_path.glob("*.jpg")) + list(image_path.glob("*.png"))

    if not images:
        print(f"No images found in {image_dir}")
        return

    # Test on first image
    test_img = images[0]
    print(f"Test Image: {test_img.name}\n")

    # Test with one standard question
    test_question = "Anatomical Structures: List all visible anatomical structures in the image and describe their appearance"

    print(f"Question: {test_question}\n")
    answer = vqa.answer_question(test_img, test_question, max_new_tokens=100)
    print(f"Answer: {answer}\n")

    print(f"{'='*60}")
    print(f"Test Complete for {category_name}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python test_vqa_category.py <category_name> <model_dir> <image_dir>")
        sys.exit(1)

    category_name = sys.argv[1]
    model_dir = sys.argv[2]
    image_dir = sys.argv[3]

    test_category_vqa(category_name, model_dir, image_dir)
