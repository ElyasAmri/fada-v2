"""Comprehensive accuracy test for VLM models on FADA dataset"""

import torch
from transformers import (
    BlipProcessor, Blip2ForConditionalGeneration,
    AutoModelForCausalLM, AutoProcessor, AutoModelForImageTextToText,
    InstructBlipProcessor, InstructBlipForConditionalGeneration,
    BitsAndBytesConfig, PaliGemmaForConditionalGeneration
)
from PIL import Image
from pathlib import Path
import time
import json
from typing import Dict, List, Tuple
import re

print("="*70)
print("VLM Accuracy Testing on FADA Dataset")
print("="*70)

# Configuration
DATA_DIR = Path("data/Fetal Ultrasound")
device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_IMAGES_PER_CATEGORY = 10  # Test on 10 images per category (50 total)

# Keywords for scoring responses
FETAL_KEYWORDS = {
    'fetal', 'fetus', 'baby', 'embryo', 'prenatal', 'gestational',
    'amniotic', 'placenta', 'umbilical', 'cord'
}

ANATOMY_KEYWORDS = {
    'Abodomen': ['abdomen', 'stomach', 'liver', 'kidney', 'bowel', 'intestine', 'bladder'],
    'Aorta': ['aorta', 'heart', 'cardiac', 'vessel', 'artery', 'ventricle', 'atrium', 'chamber'],
    'Cervical': ['cervical', 'cervix', 'neck', 'spine', 'vertebra'],
    'Cervix': ['cervix', 'cervical', 'uterus', 'vagina'],
    'Femur': ['femur', 'bone', 'leg', 'limb', 'thigh']
}

def load_test_images():
    """Load test images from all categories"""
    test_images = []
    categories = []

    for cat_dir in DATA_DIR.iterdir():
        if not cat_dir.is_dir():
            continue

        cat_name = cat_dir.name
        categories.append(cat_name)
        images = list(cat_dir.glob("*.png")) + list(cat_dir.glob("*.jpg"))

        # Take up to MAX_IMAGES_PER_CATEGORY images
        for img_path in images[:MAX_IMAGES_PER_CATEGORY]:
            test_images.append({
                'path': img_path,
                'category': cat_name
            })

    print(f"\nLoaded {len(test_images)} images from {len(categories)} categories")
    return test_images, categories

def score_response(response: str, category: str) -> Dict[str, float]:
    """Score a response based on content"""
    response_lower = response.lower()

    scores = {
        'fetal_context': 0.0,  # Does it recognize fetal/pregnancy context?
        'anatomy_correct': 0.0,  # Does it identify correct anatomical structures?
        'detail_level': 0.0,  # How detailed is the response?
        'hallucination': 0.0  # Does it mention incorrect structures?
    }

    # Check for fetal context
    fetal_matches = sum(1 for keyword in FETAL_KEYWORDS if keyword in response_lower)
    scores['fetal_context'] = min(1.0, fetal_matches / 3)  # Max score if 3+ fetal keywords

    # Check for correct anatomy
    if category in ANATOMY_KEYWORDS:
        anatomy_matches = sum(1 for keyword in ANATOMY_KEYWORDS[category] if keyword in response_lower)
        scores['anatomy_correct'] = min(1.0, anatomy_matches / 2)  # Max score if 2+ correct keywords

    # Check detail level (word count as proxy)
    word_count = len(response.split())
    if word_count < 5:
        scores['detail_level'] = 0.2
    elif word_count < 15:
        scores['detail_level'] = 0.5
    elif word_count < 30:
        scores['detail_level'] = 0.8
    else:
        scores['detail_level'] = 1.0

    # Check for hallucinations (mentioning wrong anatomical structures)
    wrong_anatomy = 0
    for other_cat, keywords in ANATOMY_KEYWORDS.items():
        if other_cat != category:
            wrong_anatomy += sum(1 for keyword in keywords if keyword in response_lower)
    scores['hallucination'] = max(0, 1.0 - (wrong_anatomy * 0.25))  # -0.25 per wrong keyword

    # Calculate overall score
    scores['overall'] = (
        scores['fetal_context'] * 0.3 +
        scores['anatomy_correct'] * 0.3 +
        scores['detail_level'] * 0.2 +
        scores['hallucination'] * 0.2
    )

    return scores

def test_blip2(test_images):
    """Test BLIP-2 model"""
    print("\n" + "="*50)
    print("Testing BLIP-2...")
    print("="*50)

    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            device_map="auto",
            torch_dtype=torch.float16
        )

        results = []
        for i, item in enumerate(test_images):
            if i % 10 == 0:
                print(f"  Processing image {i+1}/{len(test_images)}...")

            image = Image.open(item['path']).convert('RGB')
            prompt = "Question: What anatomical structures can you see in this ultrasound image? Answer:"

            inputs = processor(image, text=prompt, return_tensors="pt").to(device)

            start_time = time.time()
            generated_ids = model.generate(**inputs, max_new_tokens=100)
            response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            gen_time = time.time() - start_time

            scores = score_response(response, item['category'])

            results.append({
                'image': item['path'].name,
                'category': item['category'],
                'response': response,
                'time': gen_time,
                'scores': scores
            })

        # Clear memory
        del model, processor
        torch.cuda.empty_cache()

        return results

    except Exception as e:
        print(f"  Error testing BLIP-2: {e}")
        return []

def test_moondream2(test_images):
    """Test Moondream2 model"""
    print("\n" + "="*50)
    print("Testing Moondream2...")
    print("="*50)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision="2025-06-21",
            trust_remote_code=True,
            device_map={"": device}
        )

        results = []
        for i, item in enumerate(test_images):
            if i % 10 == 0:
                print(f"  Processing image {i+1}/{len(test_images)}...")

            image = Image.open(item['path'])
            question = "What anatomical structures can you see in this ultrasound image?"

            start_time = time.time()
            result = model.query(image, question)
            response = result["answer"]
            gen_time = time.time() - start_time

            scores = score_response(response, item['category'])

            results.append({
                'image': item['path'].name,
                'category': item['category'],
                'response': response,
                'time': gen_time,
                'scores': scores
            })

        # Clear memory
        del model
        torch.cuda.empty_cache()

        return results

    except Exception as e:
        print(f"  Error testing Moondream2: {e}")
        return []

def test_paligemma_8bit(test_images):
    """Test PaliGemma with 8-bit quantization"""
    print("\n" + "="*50)
    print("Testing PaliGemma-3B (8-bit)...")
    print("="*50)

    try:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )

        processor = AutoProcessor.from_pretrained("google/paligemma-3b-mix-224")
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            "google/paligemma-3b-mix-224",
            quantization_config=quantization_config,
            device_map="auto"
        ).eval()

        results = []
        for i, item in enumerate(test_images):
            if i % 10 == 0:
                print(f"  Processing image {i+1}/{len(test_images)}...")

            image = Image.open(item['path']).convert('RGB')
            prompt = "What anatomical structures can you see in this ultrasound image?"

            model_inputs = processor(text=prompt, images=image, return_tensors="pt")
            model_inputs = {k: v.to("cuda") for k, v in model_inputs.items()}
            input_len = model_inputs["input_ids"].shape[-1]

            start_time = time.time()
            with torch.inference_mode():
                generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
            response = processor.decode(generation[0][input_len:], skip_special_tokens=True)
            gen_time = time.time() - start_time

            scores = score_response(response, item['category'])

            results.append({
                'image': item['path'].name,
                'category': item['category'],
                'response': response,
                'time': gen_time,
                'scores': scores
            })

        # Clear memory
        del model, processor
        torch.cuda.empty_cache()

        return results

    except Exception as e:
        print(f"  Error testing PaliGemma: {e}")
        return []

def test_instructblip_4bit(test_images):
    """Test InstructBLIP with 4-bit quantization"""
    print("\n" + "="*50)
    print("Testing InstructBLIP-7B (4-bit)...")
    print("="*50)

    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
        model = InstructBlipForConditionalGeneration.from_pretrained(
            "Salesforce/instructblip-vicuna-7b",
            quantization_config=quantization_config,
            device_map="auto"
        )

        results = []
        for i, item in enumerate(test_images):
            if i % 10 == 0:
                print(f"  Processing image {i+1}/{len(test_images)}...")

            image = Image.open(item['path']).convert('RGB')
            prompt = "What anatomical structures can you see in this ultrasound image?"

            inputs = processor(images=image, text=prompt, return_tensors="pt")
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=5,
                    max_length=256,
                    min_length=1,
                    repetition_penalty=1.5
                )
            response = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            gen_time = time.time() - start_time

            scores = score_response(response, item['category'])

            results.append({
                'image': item['path'].name,
                'category': item['category'],
                'response': response,
                'time': gen_time,
                'scores': scores
            })

        # Clear memory
        del model, processor
        torch.cuda.empty_cache()

        return results

    except Exception as e:
        print(f"  Error testing InstructBLIP: {e}")
        return []

def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate accuracy metrics from results"""
    if not results:
        return {}

    metrics = {
        'total_images': len(results),
        'avg_time': sum(r['time'] for r in results) / len(results),
        'avg_scores': {},
        'per_category': {}
    }

    # Calculate average scores
    score_keys = ['fetal_context', 'anatomy_correct', 'detail_level', 'hallucination', 'overall']
    for key in score_keys:
        metrics['avg_scores'][key] = sum(r['scores'][key] for r in results) / len(results)

    # Calculate per-category scores
    categories = set(r['category'] for r in results)
    for cat in categories:
        cat_results = [r for r in results if r['category'] == cat]
        metrics['per_category'][cat] = {
            'count': len(cat_results),
            'avg_overall': sum(r['scores']['overall'] for r in cat_results) / len(cat_results)
        }

    return metrics

def main():
    """Main testing function"""
    # Load test images
    test_images, categories = load_test_images()

    # Test models
    all_results = {}

    # Test BLIP-2
    print("\nTesting BLIP-2 (baseline)...")
    all_results['BLIP-2'] = test_blip2(test_images)

    # Test Moondream2
    print("\nTesting Moondream2...")
    all_results['Moondream2'] = test_moondream2(test_images)

    # Test PaliGemma 8-bit
    print("\nTesting PaliGemma-3B (8-bit)...")
    all_results['PaliGemma-8bit'] = test_paligemma_8bit(test_images)

    # Test InstructBLIP 4-bit
    print("\nTesting InstructBLIP-7B (4-bit)...")
    all_results['InstructBLIP-4bit'] = test_instructblip_4bit(test_images)

    # Calculate metrics for each model
    print("\n" + "="*70)
    print("ACCURACY RESULTS SUMMARY")
    print("="*70)

    metrics_summary = {}
    for model_name, results in all_results.items():
        if results:
            metrics = calculate_metrics(results)
            metrics_summary[model_name] = metrics

            print(f"\n{model_name}:")
            print(f"  Images tested: {metrics['total_images']}")
            print(f"  Avg generation time: {metrics['avg_time']:.2f}s")
            print(f"  Accuracy scores (0-1 scale):")
            print(f"    - Fetal context: {metrics['avg_scores']['fetal_context']:.3f}")
            print(f"    - Anatomy correct: {metrics['avg_scores']['anatomy_correct']:.3f}")
            print(f"    - Detail level: {metrics['avg_scores']['detail_level']:.3f}")
            print(f"    - No hallucination: {metrics['avg_scores']['hallucination']:.3f}")
            print(f"    - OVERALL: {metrics['avg_scores']['overall']:.3f}")

    # Save detailed results
    with open("vlm_accuracy_results.json", "w") as f:
        json.dump({
            'test_config': {
                'images_per_category': MAX_IMAGES_PER_CATEGORY,
                'total_images': len(test_images),
                'categories': categories
            },
            'results': all_results,
            'metrics': metrics_summary
        }, f, indent=2)

    print(f"\nDetailed results saved to: vlm_accuracy_results.json")

    # Find best model
    if metrics_summary:
        best_model = max(metrics_summary.items(),
                        key=lambda x: x[1]['avg_scores']['overall'])
        print(f"\n🏆 BEST MODEL: {best_model[0]} (Overall score: {best_model[1]['avg_scores']['overall']:.3f})")

if __name__ == "__main__":
    main()