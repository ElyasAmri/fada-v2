"""
VLM Evaluation Pipeline: Compare fine-tuned Qwen2.5-VL-7B against Gemini ground truth.

Uses embedding similarity to measure how well the fine-tuned model matches
the Gemini annotations used for training.

Usage:
    # Full evaluation
    python experiments/evaluation/evaluate_vlm.py

    # With custom test subset
    python experiments/evaluation/evaluate_vlm.py --test-data outputs/evaluation/test_subset.jsonl

    # Inference only (for vast.ai - saves predictions for later scoring)
    python experiments/evaluation/evaluate_vlm.py --inference-only

    # Scoring only (use cached predictions)
    python experiments/evaluation/evaluate_vlm.py --score-only --predictions outputs/evaluation/predictions.jsonl
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from .config import (
    BASE_MODEL_ID,
    ADAPTER_PATH,
    STRATIFIED_TEST_DATA,
    OUTPUTS_DIR,
    SYSTEM_PROMPT,
    MAX_NEW_TOKENS,
    GENERATION_TEMPERATURE,
    DEFAULT_EMBEDDING_MODEL,
)


def extract_category_from_path(image_path: str) -> str:
    """Extract category name from image path."""
    parts = Path(image_path).parts
    for i, part in enumerate(parts):
        if part == "Fetal Ultrasound":
            return parts[i + 1]
    return "Unknown"


def load_test_samples(test_path: Path) -> List[Dict]:
    """Load test samples from JSONL file."""
    samples = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


class VLMEvaluator:
    """Evaluate fine-tuned VLM against ground truth annotations."""

    def __init__(
        self,
        adapter_path: Path = ADAPTER_PATH,
        use_4bit: bool = True,
        device_map: str = "auto"
    ):
        self.adapter_path = adapter_path
        self.use_4bit = use_4bit
        self.device_map = device_map
        self.model = None
        self.processor = None

    def load_model(self):
        """Load fine-tuned model with LoRA adapter."""
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
        from peft import PeftModel

        print(f"Loading base model: {BASE_MODEL_ID}")

        self.processor = AutoProcessor.from_pretrained(
            BASE_MODEL_ID,
            trust_remote_code=True,
        )

        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            bnb_config = None

        model = AutoModelForImageTextToText.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=bnb_config,
            device_map=self.device_map,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # Load LoRA adapter
        print(f"Loading adapter from: {self.adapter_path}")
        if not self.adapter_path.exists():
            raise FileNotFoundError(f"Adapter not found at {self.adapter_path}")
        self.model = PeftModel.from_pretrained(model, str(self.adapter_path))
        self.model.eval()

        print("Model loaded successfully")

    def generate_response(
        self,
        image_path: str,
        question: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = GENERATION_TEMPERATURE
    ) -> str:
        """Generate response for a single image-question pair."""
        import torch
        from PIL import Image

        image = Image.open(image_path).convert('RGB')

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=text,
            images=[image],
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

        input_len = inputs['input_ids'].shape[1]
        response = self.processor.tokenizer.decode(
            output_ids[0][input_len:],
            skip_special_tokens=True
        )

        return response.strip()

    def run_inference(
        self,
        test_samples: List[Dict],
        checkpoint_interval: int = 50,
        checkpoint_path: Optional[Path] = None
    ) -> List[Dict]:
        """
        Run inference on all test samples.

        Each sample has format:
        {
            "messages": [system, user, assistant],
            "images": ["path/to/image.png"]
        }

        Returns list of:
        {
            "sample_id": int,
            "image_path": str,
            "category": str,
            "question": str,
            "ground_truth": str,
            "prediction": str
        }
        """
        results = []

        for idx, sample in enumerate(tqdm(test_samples, desc="Running inference")):
            image_path = sample['images'][0]
            messages = sample['messages']

            # Extract question from user message
            user_msg = messages[1]  # Second message is user
            question = None
            for content in user_msg['content']:
                if isinstance(content, dict) and content.get('type') == 'text':
                    question = content['text']
                    break

            # Extract ground truth from assistant message
            ground_truth = messages[2]['content']  # Third message is assistant

            # Extract category from path
            category = extract_category_from_path(image_path)

            # Generate prediction
            try:
                prediction = self.generate_response(image_path, question)
            except Exception as e:
                print(f"\nError processing {image_path}: {e}")
                prediction = ""

            results.append({
                "sample_id": idx,
                "image_path": image_path,
                "category": category,
                "question": question[:100] + "..." if len(question) > 100 else question,
                "ground_truth": ground_truth,
                "prediction": prediction
            })

            # Periodic checkpoint save
            if checkpoint_path and (idx + 1) % checkpoint_interval == 0:
                self._save_checkpoint(results, checkpoint_path)
                print(f"\n  Checkpoint saved: {idx + 1}/{len(test_samples)} samples")

        return results

    def _save_checkpoint(self, results: List[Dict], checkpoint_path: Path):
        """Save intermediate results to checkpoint file."""
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')


def compute_scores(
    results: List[Dict],
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    device: str = "cpu"
) -> Dict:
    """Compute embedding similarity scores."""
    from embedding_scorer import EmbeddingScorer

    predictions = [r['prediction'] for r in results]
    ground_truths = [r['ground_truth'] for r in results]
    categories = [r['category'] for r in results]

    # Initialize scorer (use CPU by default to avoid CUDA compatibility issues)
    scorer = EmbeddingScorer(model_name=embedding_model, device=device)

    # Compute similarities
    similarities = scorer.compute_similarity(predictions, ground_truths)

    # Aggregate metrics
    aggregate = scorer.compute_aggregate_metrics(similarities)

    # Per-category breakdown
    category_metrics = scorer.compute_category_metrics(similarities, categories)

    # Sample-level scores
    sample_scores = [
        {
            "sample_id": r['sample_id'],
            "category": r['category'],
            "similarity": float(similarities[i])
        }
        for i, r in enumerate(results)
    ]

    return {
        "aggregate": aggregate,
        "per_category": category_metrics,
        "sample_scores": sample_scores
    }


def print_results_summary(scores: Dict):
    """Print formatted results summary."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    agg = scores['aggregate']
    print(f"\nOverall Similarity Score: {agg['mean_similarity']:.4f}")
    print(f"Standard Deviation:       {agg['std_similarity']:.4f}")
    print(f"Median:                   {agg['median_similarity']:.4f}")
    print(f"Range:                    [{agg['min_similarity']:.4f}, {agg['max_similarity']:.4f}]")
    print(f"25th Percentile:          {agg['percentile_25']:.4f}")
    print(f"75th Percentile:          {agg['percentile_75']:.4f}")
    print(f"Number of Samples:        {agg['num_samples']}")

    print("\nPer-Category Breakdown:")
    print("-" * 60)
    for category, cat_scores in sorted(scores['per_category'].items()):
        mean = cat_scores['mean_similarity']
        std = cat_scores['std_similarity']
        n = cat_scores['num_samples']
        print(f"  {category:35s}: {mean:.4f} (+/- {std:.4f})  n={n}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned VLM against ground truth")

    parser.add_argument(
        '--test-data', type=str, default=str(STRATIFIED_TEST_DATA),
        help='Path to test subset JSONL'
    )
    parser.add_argument(
        '--inference-only', action='store_true',
        help='Only run inference, skip scoring (for vast.ai)'
    )
    parser.add_argument(
        '--score-only', action='store_true',
        help='Only run scoring on existing predictions'
    )
    parser.add_argument(
        '--predictions', type=str, default=None,
        help='Path to predictions JSONL (for --score-only)'
    )
    parser.add_argument(
        '--embedding-model', type=str, default=DEFAULT_EMBEDDING_MODEL,
        choices=['all-mpnet-base-v2', 'all-MiniLM-L6-v2'],
        help='Sentence transformer model for embedding similarity'
    )
    parser.add_argument(
        '--no-4bit', action='store_true',
        help='Disable 4-bit quantization (requires more VRAM)'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Custom output path for results JSON'
    )

    args = parser.parse_args()

    # Setup output directory
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Print configuration
    print("=" * 70)
    print("VLM Evaluation Pipeline")
    print("=" * 70)
    print(f"Test data:       {args.test_data}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Mode:            {'inference-only' if args.inference_only else 'score-only' if args.score_only else 'full'}")

    results = None

    if not args.score_only:
        # Check test data exists
        test_path = Path(args.test_data)
        if not test_path.exists():
            print(f"\nError: Test data not found: {test_path}")
            print(f"Run create_test_subset.py first to generate the test subset.")
            return 1

        # Load test samples
        test_samples = load_test_samples(test_path)
        print(f"\nLoaded {len(test_samples)} test samples")

        # Check CUDA availability
        try:
            import torch
            if torch.cuda.is_available():
                print(f"CUDA: {torch.cuda.get_device_name(0)}")
                print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                print("WARNING: CUDA not available, inference will be slow!")
        except ImportError:
            print("WARNING: PyTorch not available")

        # Initialize evaluator
        evaluator = VLMEvaluator(use_4bit=not args.no_4bit)
        evaluator.load_model()

        # Run inference
        checkpoint_path = OUTPUTS_DIR / f"predictions_checkpoint_{timestamp}.jsonl"
        results = evaluator.run_inference(test_samples, checkpoint_path=checkpoint_path)

        # Save predictions
        predictions_path = OUTPUTS_DIR / f"predictions_{timestamp}.jsonl"
        with open(predictions_path, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f"\nPredictions saved to: {predictions_path}")

        # Clean up checkpoint if we completed successfully
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    if args.inference_only:
        print("\nInference complete. Skipping scoring.")
        print("To score later, run with --score-only --predictions <path>")
        return 0

    # Load predictions if score-only mode
    if args.score_only:
        if not args.predictions:
            print("\nError: --predictions required when using --score-only")
            return 1

        predictions_path = Path(args.predictions)
        if not predictions_path.exists():
            print(f"\nError: Predictions file not found: {predictions_path}")
            return 1

        results = []
        with open(predictions_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        print(f"\nLoaded {len(results)} predictions from {predictions_path}")

    # Compute scores
    print("\nComputing embedding similarity scores...")
    scores = compute_scores(results, embedding_model=args.embedding_model)

    # Build final output
    final_results = {
        "metadata": {
            "timestamp": timestamp,
            "test_data": args.test_data,
            "embedding_model": args.embedding_model,
            "base_model": BASE_MODEL_ID,
            "adapter_path": str(ADAPTER_PATH),
            "num_samples": len(results),
        },
        "scores": scores
    }

    # Save results
    output_path = Path(args.output) if args.output else OUTPUTS_DIR / f"evaluation_results_{timestamp}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    # Print summary
    print_results_summary(scores)

    print(f"\nResults saved to: {output_path}")
    return 0


if __name__ == "__main__":
    exit(main())
