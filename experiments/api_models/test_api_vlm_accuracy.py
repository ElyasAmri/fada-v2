"""
API VLM Accuracy Test - Evaluate Gemini and Grok vision models on fetal ultrasound images
Tests all 8 clinical questions per image and tracks accuracy metrics

Usage:
    python test_api_vlm_accuracy.py --models gemini grok --images-per-category 5
    python test_api_vlm_accuracy.py --models gemini --images-per-category 10 --output-dir results/my_test
"""
import sys
# Ensure output is not buffered
sys.stdout.reconfigure(line_buffering=True)
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from PIL import Image
from tqdm import tqdm

from src.data.question_loader import QuestionLoader
from src.inference.gemini_vlm import GeminiVLM, create_gemini_vlm
from src.inference.grok_vlm import GrokVLM, create_grok_vlm


def get_test_images(
    question_loader: QuestionLoader,
    images_per_category: int = 5,
    categories: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Get test images from the dataset

    Args:
        question_loader: QuestionLoader instance
        images_per_category: Number of images per category to test
        categories: Specific categories to test (None = all available)

    Returns:
        List of dicts with 'path', 'category', 'name' keys
    """
    test_images = []

    available_categories = question_loader.get_categories()
    if categories:
        available_categories = [c for c in categories if c in available_categories]

    for category in available_categories:
        images = question_loader.get_category_images(category)
        # Take first N images
        for img_path in images[:images_per_category]:
            test_images.append({
                'path': img_path,
                'category': category,
                'name': img_path.name
            })

    return test_images


def evaluate_response(response: str, category: str, question_idx: int) -> Dict[str, Any]:
    """
    Evaluate a model response for accuracy

    Args:
        response: Model's response text
        category: Image category (e.g., "Femur", "Brain")
        question_idx: Question index (0-7)

    Returns:
        Dict with evaluation metrics
    """
    response_lower = response.lower()

    # Fetal context keywords
    fetal_keywords = ['fetal', 'fetus', 'ultrasound', 'pregnancy', 'prenatal',
                      'gestational', 'amniotic', 'placenta', 'umbilical']

    # Category-specific anatomy keywords
    anatomy_keywords = {
        'Abodomen': ['abdomen', 'stomach', 'liver', 'kidney', 'bowel', 'intestine', 'bladder', 'abdominal'],
        'Aorta': ['aorta', 'heart', 'cardiac', 'vessel', 'artery', 'ventricle', 'atrium'],
        'Brain': ['brain', 'skull', 'cerebral', 'ventricle', 'head', 'cranial', 'hemisphere'],
        'Femur': ['femur', 'bone', 'leg', 'limb', 'thigh', 'long bone', 'skeletal'],
        'Heart': ['heart', 'cardiac', 'ventricle', 'atrium', 'chamber', 'valve'],
        'Thorax': ['thorax', 'chest', 'lung', 'rib', 'diaphragm', 'thoracic'],
        'Cervical': ['cervical', 'cervix', 'neck', 'spine'],
        'Cervix': ['cervix', 'cervical', 'uterus'],
    }

    # Check for fetal context
    has_fetal_context = any(kw in response_lower for kw in fetal_keywords)

    # Check for correct anatomy
    category_kws = anatomy_keywords.get(category, [category.lower()])
    has_correct_anatomy = any(kw in response_lower for kw in category_kws)

    # Check response quality based on word count
    # Thresholds based on typical medical response lengths:
    # - < 5 words: Too brief to be useful (0.2 score)
    # - 5-14 words: Basic response, missing context (0.5 score)
    # - 15-29 words: Good response with some detail (0.8 score)
    # - 30+ words: Detailed response with full explanation (1.0 score)
    word_count = len(response.split())
    if word_count < 5:
        detail_score = 0.2
    elif word_count < 15:
        detail_score = 0.5
    elif word_count < 30:
        detail_score = 0.8
    else:
        detail_score = 1.0

    # Check for hallucination (mentioning wrong anatomy)
    # Penalty of 0.1 per wrong anatomy mentioned strongly
    # Only checks first 2 keywords per category (most specific ones)
    hallucination_penalty = 0
    wrong_categories = [cat for cat in anatomy_keywords.keys() if cat != category]
    for wrong_cat in wrong_categories:
        wrong_kws = anatomy_keywords.get(wrong_cat, [])
        # Only penalize if strongly claiming wrong anatomy (not just mentioning)
        if any(kw in response_lower and category.lower() not in response_lower for kw in wrong_kws[:2]):
            hallucination_penalty += 0.1

    # Calculate overall score using weighted components:
    # - Fetal context (30%): Does response mention fetal/pregnancy context?
    # - Correct anatomy (40%): Does response correctly identify the anatomy type?
    # - Detail score (20%): How detailed is the response?
    # - Hallucination penalty (10%): Deduction for mentioning wrong anatomy
    overall_score = (
        (1.0 if has_fetal_context else 0.0) * 0.3 +
        (1.0 if has_correct_anatomy else 0.0) * 0.4 +
        detail_score * 0.2 +
        max(0, 1.0 - hallucination_penalty) * 0.1
    )

    return {
        'has_fetal_context': has_fetal_context,
        'has_correct_anatomy': has_correct_anatomy,
        'word_count': word_count,
        'detail_score': detail_score,
        'hallucination_penalty': hallucination_penalty,
        'overall_score': overall_score
    }


def run_single_image_test(
    model,
    image_path: Path,
    questions: List[str],
    category: str
) -> Dict[str, Any]:
    """
    Run all 8 questions on a single image

    Args:
        model: VLM model instance
        image_path: Path to image file
        questions: List of 8 questions
        category: Image category

    Returns:
        Dict with results for all questions
    """
    # Load image
    image = Image.open(image_path).convert('RGB')

    results = {
        'image': image_path.name,
        'category': category,
        'questions': []
    }

    for q_idx, question in enumerate(questions):
        start_time = time.time()

        try:
            response = model.answer_question(image, question)
            elapsed_time = time.time() - start_time

            # Evaluate response
            evaluation = evaluate_response(response, category, q_idx)

            results['questions'].append({
                'question_idx': q_idx,
                'question': question[:50] + '...' if len(question) > 50 else question,
                'response': response,
                'time': elapsed_time,
                'evaluation': evaluation
            })

        except Exception as e:
            elapsed_time = time.time() - start_time
            results['questions'].append({
                'question_idx': q_idx,
                'question': question[:50] + '...' if len(question) > 50 else question,
                'response': f"ERROR: {str(e)}",
                'time': elapsed_time,
                'evaluation': {
                    'has_fetal_context': False,
                    'has_correct_anatomy': False,
                    'word_count': 0,
                    'detail_score': 0,
                    'hallucination_penalty': 0,
                    'overall_score': 0
                }
            })

    return results


def run_evaluation(
    models: List[Dict[str, Any]],
    images_per_category: int = 5,
    single_image_first: bool = True,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Run full evaluation across models and images

    Args:
        models: List of model configs with 'name', 'instance' keys
        images_per_category: Number of images per category
        single_image_first: If True, test single image first before full run
        output_dir: Directory to save results

    Returns:
        Dict with all results
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)

    # Initialize question loader
    question_loader = QuestionLoader(str(project_root / 'data' / 'Fetal Ultrasound'))
    questions = question_loader.get_questions()
    question_names = question_loader.get_question_short_names()

    print(f"Loaded {len(questions)} questions")
    print(f"Available categories: {question_loader.get_categories()}")

    # Get test images
    test_images = get_test_images(question_loader, images_per_category)
    print(f"Total test images: {len(test_images)}")

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'images_per_category': images_per_category,
            'total_images': len(test_images),
            'questions': question_names
        },
        'models': {}
    }

    for model_config in models:
        model_name = model_config['name']
        model = model_config['instance']

        print(f"\n{'='*60}")
        print(f"Testing: {model_name}")
        print(f"{'='*60}")

        try:
            model.load()
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            all_results['models'][model_name] = {'error': str(e)}
            continue

        model_results = []

        # Single image test first
        if single_image_first and test_images:
            print("\n--- Single Image Test (all 8 questions) ---")
            first_image = test_images[0]
            result = run_single_image_test(
                model,
                first_image['path'],
                questions,
                first_image['category']
            )
            model_results.append(result)

            # Print single image results
            print(f"\nImage: {first_image['name']} ({first_image['category']})")
            for q_result in result['questions']:
                score = q_result['evaluation']['overall_score']
                print(f"  Q{q_result['question_idx']+1}: {score:.2f} ({q_result['time']:.2f}s)")

            # Ask user to continue
            avg_score = sum(q['evaluation']['overall_score'] for q in result['questions']) / len(result['questions'])
            print(f"\nAverage score: {avg_score:.2%}")

        # Full evaluation
        print(f"\n--- Full Evaluation ({len(test_images)} images) ---")
        for img_info in tqdm(test_images[1:] if single_image_first else test_images,
                            desc=model_name):
            result = run_single_image_test(
                model,
                img_info['path'],
                questions,
                img_info['category']
            )
            model_results.append(result)

        # Calculate aggregate metrics
        all_scores = []
        per_category_scores = {}
        per_question_scores = {i: [] for i in range(8)}

        for result in model_results:
            category = result['category']
            if category not in per_category_scores:
                per_category_scores[category] = []

            for q_result in result['questions']:
                score = q_result['evaluation']['overall_score']
                all_scores.append(score)
                per_category_scores[category].append(score)
                per_question_scores[q_result['question_idx']].append(score)

        all_results['models'][model_name] = {
            'results': model_results,
            'metrics': {
                'overall_accuracy': sum(all_scores) / len(all_scores) if all_scores else 0,
                'total_images': len(model_results),
                'per_category': {
                    cat: sum(scores) / len(scores) if scores else 0
                    for cat, scores in per_category_scores.items()
                },
                'per_question': {
                    question_names[i]: sum(scores) / len(scores) if scores else 0
                    for i, scores in per_question_scores.items()
                }
            }
        }

        print(f"\n{model_name} Results:")
        print(f"  Overall Accuracy: {all_results['models'][model_name]['metrics']['overall_accuracy']:.2%}")

        model.unload()

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'api_vlm_results_{timestamp}.json'

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return all_results


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Evaluate API VLM models on fetal ultrasound images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --models gemini grok --images-per-category 5
  %(prog)s --models gemini --images-per-category 10 --output-dir results/custom
  %(prog)s --gemini-model gemini-2.5-flash --grok-model grok-4
        """
    )

    parser.add_argument(
        "--models", "-m",
        nargs="+",
        choices=["gemini", "grok", "all"],
        default=["all"],
        help="Models to evaluate (default: all)"
    )

    parser.add_argument(
        "--images-per-category", "-n",
        type=int,
        default=5,
        help="Number of images per category to test (default: 5)"
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Output directory for results (default: experiments/api_models/results)"
    )

    parser.add_argument(
        "--gemini-model",
        default="gemini-3-pro-preview",
        help="Gemini model to use (default: gemini-3-pro-preview)"
    )

    parser.add_argument(
        "--grok-model",
        default="grok-4",
        help="Grok model to use (default: grok-4)"
    )

    parser.add_argument(
        "--thinking-level",
        choices=["none", "low", "medium", "high"],
        default="low",
        help="Gemini thinking level (default: low)"
    )

    parser.add_argument(
        "--skip-single-test",
        action="store_true",
        help="Skip single image test and run full evaluation immediately"
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    print("API VLM Accuracy Test")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Models: {args.models}")
    print(f"  Images per category: {args.images_per_category}")
    print(f"  Gemini model: {args.gemini_model}")
    print(f"  Grok model: {args.grok_model}")
    print("=" * 60)

    # Determine which models to run
    run_gemini = "all" in args.models or "gemini" in args.models
    run_grok = "all" in args.models or "grok" in args.models

    # Initialize models
    models = []

    # Try Gemini
    if run_gemini:
        try:
            gemini = create_gemini_vlm(model=args.gemini_model, thinking_level=args.thinking_level)
            models.append({'name': f'Gemini ({args.gemini_model})', 'instance': gemini})
            print(f"[OK] Gemini initialized: {args.gemini_model}")
        except Exception as e:
            print(f"[SKIP] Gemini: {e}")

    # Try Grok
    if run_grok:
        try:
            grok = create_grok_vlm(model=args.grok_model)
            models.append({'name': f'Grok ({args.grok_model})', 'instance': grok})
            print(f"[OK] Grok initialized: {args.grok_model}")
        except Exception as e:
            print(f"[SKIP] Grok: {e}")

    if not models:
        print("\nNo models available. Please check API keys in .env.local")
        print("Required: GEMINI_API_KEY and/or XAI_API_KEY")
        return

    # Run evaluation
    results = run_evaluation(
        models=models,
        images_per_category=args.images_per_category,
        single_image_first=not args.skip_single_test,
        output_dir=args.output_dir
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for model_name, model_data in results['models'].items():
        if 'error' in model_data:
            print(f"{model_name}: ERROR - {model_data['error']}")
        else:
            metrics = model_data['metrics']
            print(f"\n{model_name}:")
            print(f"  Overall Accuracy: {metrics['overall_accuracy']:.2%}")
            print(f"  Per Category:")
            for cat, score in metrics['per_category'].items():
                print(f"    {cat}: {score:.2%}")


if __name__ == '__main__':
    main()
