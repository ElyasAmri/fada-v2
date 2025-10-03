"""Fast accuracy test for top VLM models on FADA dataset (5 images per category)"""

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig, PaliGemmaForConditionalGeneration
from PIL import Image
from pathlib import Path
import time
import json

print("="*70)
print("VLM Accuracy Testing on FADA Dataset (Fast Version)")
print("="*70)

# Configuration
DATA_DIR = Path("data/Fetal Ultrasound")
device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_IMAGES_PER_CATEGORY = 5  # Test on 5 images per category

# Keywords for scoring
FETAL_KEYWORDS = ['fetal', 'fetus', 'baby', 'embryo', 'prenatal', 'placenta', 'umbilical']
MATERNAL_KEYWORDS = ['uterus', 'vagina', 'fallopian', 'ovary', 'ovaries', 'maternal']

def score_response(response, category):
    """Simple scoring based on keywords"""
    response_lower = response.lower()

    scores = {
        'has_fetal_context': any(kw in response_lower for kw in FETAL_KEYWORDS),
        'has_maternal_context': any(kw in response_lower for kw in MATERNAL_KEYWORDS),
        'mentions_ultrasound': 'ultrasound' in response_lower,
        'response_length': len(response.split()),
        'category': category
    }

    # Anatomy check
    anatomy_correct = False
    if 'abodomen' in category.lower() or 'abdomen' in category.lower():
        anatomy_correct = any(w in response_lower for w in ['abdomen', 'stomach', 'liver', 'kidney'])
    elif 'aorta' in category.lower():
        anatomy_correct = any(w in response_lower for w in ['aorta', 'heart', 'cardiac', 'vessel'])
    elif 'cervix' in category.lower() or 'cervical' in category.lower():
        anatomy_correct = any(w in response_lower for w in ['cervix', 'cervical', 'neck'])
    elif 'femur' in category.lower():
        anatomy_correct = any(w in response_lower for w in ['femur', 'bone', 'leg', 'limb'])

    scores['anatomy_correct'] = anatomy_correct
    return scores

def load_test_images():
    """Load 5 images per category"""
    test_images = []
    for cat_dir in DATA_DIR.iterdir():
        if not cat_dir.is_dir():
            continue
        cat_name = cat_dir.name
        images = list(cat_dir.glob("*.png")) + list(cat_dir.glob("*.jpg"))
        for img_path in images[:MAX_IMAGES_PER_CATEGORY]:
            test_images.append({'path': img_path, 'category': cat_name})
    print(f"Loaded {len(test_images)} images from {len(set(i['category'] for i in test_images))} categories")
    return test_images

def test_moondream2():
    """Test Moondream2"""
    print("\n" + "="*50)
    print("Testing Moondream2...")

    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-06-21",
        trust_remote_code=True,
        device_map={"": device}
    )

    test_images = load_test_images()
    results = []

    for i, item in enumerate(test_images):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(test_images)} images...")

        image = Image.open(item['path'])
        question = "What anatomical structures can you see in this ultrasound image?"

        start = time.time()
        response = model.query(image, question)["answer"]
        gen_time = time.time() - start

        scores = score_response(response, item['category'])
        results.append({
            'image': item['path'].name,
            'category': item['category'],
            'response': response[:200],  # Truncate for display
            'time': gen_time,
            'scores': scores
        })

    del model
    torch.cuda.empty_cache()
    return results

def test_paligemma_8bit():
    """Test PaliGemma 8-bit"""
    print("\n" + "="*50)
    print("Testing PaliGemma-3B (8-bit)...")

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    processor = AutoProcessor.from_pretrained("google/paligemma-3b-mix-224")
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        "google/paligemma-3b-mix-224",
        quantization_config=quantization_config,
        device_map="auto"
    ).eval()

    test_images = load_test_images()
    results = []

    for i, item in enumerate(test_images):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(test_images)} images...")

        image = Image.open(item['path']).convert('RGB')
        prompt = "What anatomical structures can you see in this ultrasound image?"

        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        start = time.time()
        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        response = processor.decode(generation[0][input_len:], skip_special_tokens=True)
        gen_time = time.time() - start

        scores = score_response(response, item['category'])
        results.append({
            'image': item['path'].name,
            'category': item['category'],
            'response': response[:200],
            'time': gen_time,
            'scores': scores
        })

    del model, processor
    torch.cuda.empty_cache()
    return results

def calculate_accuracy(results):
    """Calculate accuracy metrics"""
    total = len(results)
    if total == 0:
        return {}

    metrics = {
        'total_images': total,
        'avg_time': sum(r['time'] for r in results) / total,
        'fetal_context_rate': sum(1 for r in results if r['scores']['has_fetal_context']) / total,
        'maternal_context_rate': sum(1 for r in results if r['scores']['has_maternal_context']) / total,
        'anatomy_correct_rate': sum(1 for r in results if r['scores']['anatomy_correct']) / total,
        'avg_response_length': sum(r['scores']['response_length'] for r in results) / total
    }

    # Per-category accuracy
    categories = set(r['category'] for r in results)
    metrics['per_category'] = {}
    for cat in categories:
        cat_results = [r for r in results if r['category'] == cat]
        if cat_results:
            metrics['per_category'][cat] = {
                'count': len(cat_results),
                'anatomy_correct': sum(1 for r in cat_results if r['scores']['anatomy_correct']) / len(cat_results)
            }

    return metrics

def main():
    """Run accuracy tests"""
    all_results = {}

    # Test Moondream2
    try:
        all_results['Moondream2'] = test_moondream2()
    except Exception as e:
        print(f"  Error testing Moondream2: {e}")

    # Test PaliGemma
    try:
        all_results['PaliGemma-8bit'] = test_paligemma_8bit()
    except Exception as e:
        print(f"  Error testing PaliGemma: {e}")

    # Calculate and display metrics
    print("\n" + "="*70)
    print("ACCURACY RESULTS")
    print("="*70)

    for model_name, results in all_results.items():
        if results:
            metrics = calculate_accuracy(results)
            print(f"\n{model_name}:")
            print(f"  Images tested: {metrics['total_images']}")
            print(f"  Avg time: {metrics['avg_time']:.2f}s")
            print(f"  Accuracy metrics:")
            print(f"    - Recognizes fetal context: {metrics['fetal_context_rate']:.1%}")
            print(f"    - Mentions maternal anatomy: {metrics['maternal_context_rate']:.1%}")
            print(f"    - Correct anatomy identified: {metrics['anatomy_correct_rate']:.1%}")
            print(f"    - Avg response length: {metrics['avg_response_length']:.0f} words")

            # Show per-category
            print(f"  Per-category anatomy accuracy:")
            for cat, cat_metrics in sorted(metrics['per_category'].items()):
                print(f"    - {cat}: {cat_metrics['anatomy_correct']:.1%} ({cat_metrics['count']} images)")

    # Save results
    with open("vlm_accuracy_fast_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to: vlm_accuracy_fast_results.json")

    # Determine winner
    if all_results:
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)

        for model_name, results in all_results.items():
            if results:
                metrics = calculate_accuracy(results)
                overall_score = (
                    metrics['fetal_context_rate'] * 0.4 +  # Most important
                    metrics['anatomy_correct_rate'] * 0.4 +  # Also important
                    (1 - metrics['maternal_context_rate']) * 0.2  # Penalty for maternal confusion
                )
                print(f"{model_name}: Overall Score = {overall_score:.1%}")

if __name__ == "__main__":
    main()