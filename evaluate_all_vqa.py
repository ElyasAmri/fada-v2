"""Evaluate all trained VQA models comprehensively"""
import sys
from pathlib import Path
from PIL import Image
from src.models.vqa_model import UltrasoundVQA
import json
import time

# Category configurations
CATEGORIES = {
    "Non_standard_NT": {
        "model": "outputs/blip2_1epoch/final_model",
        "images": "data/Fetal Ultrasound/Non_standard_NT"
    },
    "Abdomen": {
        "model": "outputs/blip2_abdomen/final_model",
        "images": "data/Fetal Ultrasound/Abodomen"
    },
    "Femur": {
        "model": "outputs/blip2_femur/final_model",
        "images": "data/Fetal Ultrasound/Femur"
    },
    "Thorax": {
        "model": "outputs/blip2_thorax/final_model",
        "images": "data/Fetal Ultrasound/Thorax"
    },
    "Standard_NT": {
        "model": "outputs/blip2_standard_nt/final_model",
        "images": "data/Fetal Ultrasound/Standard_NT"
    },
}

# Test questions
TEST_QUESTIONS = [
    "Anatomical Structures: List all visible anatomical structures in the image",
    "Image Quality: Evaluate the overall image quality and clarity",
    "Normality/Abnormality: Identify any visible abnormalities or deviations from normal anatomy",
]

def evaluate_category(category_name, config):
    """Evaluate VQA model for one category"""
    print(f"\n{'='*70}")
    print(f"Category: {category_name}")
    print(f"{'='*70}")

    model_path = Path(config["model"])
    if not model_path.exists():
        print(f"WARNING: Model not found: {model_path}")
        return None

    # Load model
    print(f"Loading model from {model_path}")
    vqa = UltrasoundVQA(model_path=str(model_path))
    vqa.load_model()

    # Get test images
    image_dir = Path(config["images"])
    images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

    if not images:
        print(f"WARNING: No images found in {image_dir}")
        return None

    # Test on first 3 images
    results = {
        "category": category_name,
        "model_path": str(model_path),
        "num_test_images": min(3, len(images)),
        "tests": []
    }

    for i, img_path in enumerate(images[:3]):
        print(f"\nImage {i+1}: {img_path.name}")
        image = Image.open(img_path).convert('RGB')

        test_result = {
            "image": img_path.name,
            "questions": []
        }

        for question in TEST_QUESTIONS:
            print(f"\nQ: {question}")
            start_time = time.time()
            answer = vqa.answer_question(image, question, max_new_tokens=100)
            inference_time = time.time() - start_time

            print(f"A: {answer}")
            print(f"Time: {inference_time:.2f}s")

            test_result["questions"].append({
                "question": question,
                "answer": answer,
                "inference_time": inference_time
            })

        results["tests"].append(test_result)

    # Unload model to free memory
    vqa.unload_model()

    return results

def main():
    """Run comprehensive evaluation on all categories"""
    print("="*70)
    print("VQA Model Comprehensive Evaluation")
    print("="*70)
    print(f"Testing {len(CATEGORIES)} categories")
    print(f"Questions per image: {len(TEST_QUESTIONS)}")
    print(f"Images per category: 3")

    all_results = []

    for category_name, config in CATEGORIES.items():
        result = evaluate_category(category_name, config)
        if result:
            all_results.append(result)

    # Save results
    output_file = Path("outputs/vqa_evaluation_results.json")
    output_file.parent.mkdir(exist_ok=True, parents=True)

    with open(output_file, 'w') as f:
        json.dump({
            "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "categories_tested": len(all_results),
            "total_tests": sum(r["num_test_images"] for r in all_results) * len(TEST_QUESTIONS),
            "results": all_results
        }, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Evaluation complete!")
    print(f"Results saved to: {output_file}")
    print(f"Categories tested: {len(all_results)}/{len(CATEGORIES)}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
