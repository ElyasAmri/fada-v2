#!/bin/bash
# Vast.ai VLM Evaluation Setup Script
# Run this after SSH into your Vast.ai instance
#
# Usage:
#   1. SSH into instance: ssh -p PORT root@IP
#   2. Run: bash vastai_eval_setup.sh
#
# Prerequisites:
#   - Upload test data: scp -P PORT outputs/evaluation/test_subset.jsonl root@IP:/workspace/fada-eval/data/
#   - Upload LoRA adapter: scp -P PORT -r models/qwen25vl7b_finetuned/final root@IP:/workspace/fada-eval/adapter/

set -e

echo "=============================================="
echo "  VLM Evaluation Setup for Vast.ai"
echo "=============================================="
echo ""

# 1. Check GPU
echo "[1/5] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# 2. Install dependencies
echo "[2/5] Installing dependencies..."
pip install -q --upgrade pip
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -q transformers>=4.45.0 accelerate bitsandbytes peft
pip install -q pillow tqdm

echo "Dependencies installed."
echo ""

# 3. Setup workspace
echo "[3/5] Setting up workspace..."
WORKSPACE="/workspace/fada-eval"
mkdir -p $WORKSPACE/data
mkdir -p $WORKSPACE/adapter
mkdir -p $WORKSPACE/outputs
cd $WORKSPACE

# 4. Create evaluation script
echo "[4/5] Creating evaluation script..."
cat > $WORKSPACE/run_inference.py << 'EVAL_SCRIPT'
"""
VLM Inference Script for Vast.ai
Generates predictions for evaluation test subset.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel


# Configuration
BASE_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
SYSTEM_PROMPT = """You are an expert fetal ultrasound analyst. Analyze the ultrasound image and answer the question thoroughly based on what you observe. Focus on anatomical structures, image quality, and any clinically relevant findings."""


def load_model(adapter_path: str, use_4bit: bool = True):
    """Load base model with LoRA adapter."""
    print(f"Loading base model: {BASE_MODEL_ID}")

    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
    )

    if use_4bit:
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
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    print("Model loaded successfully")
    return model, processor


def extract_category_from_path(image_path: str) -> str:
    """Extract category name from image path."""
    parts = Path(image_path).parts
    for i, part in enumerate(parts):
        if part == "Fetal Ultrasound":
            return parts[i + 1]
    return "Unknown"


def generate_response(
    model, processor, image_path: str, question: str,
    max_new_tokens: int = 512, temperature: float = 0.1
) -> str:
    """Generate response for a single image-question pair."""
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

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=text,
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    input_len = inputs['input_ids'].shape[1]
    response = processor.tokenizer.decode(
        output_ids[0][input_len:],
        skip_special_tokens=True
    )

    return response.strip()


def main():
    parser = argparse.ArgumentParser(description="Run VLM inference for evaluation")
    parser.add_argument('--test-data', type=str, default='data/test_subset.jsonl')
    parser.add_argument('--adapter', type=str, default='adapter')
    parser.add_argument('--output', type=str, default='outputs')
    parser.add_argument('--no-4bit', action='store_true')
    parser.add_argument('--checkpoint-interval', type=int, default=50)
    args = parser.parse_args()

    # Setup paths
    test_path = Path(args.test_data)
    adapter_path = Path(args.adapter)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("VLM Inference for Evaluation")
    print("=" * 60)
    print(f"Test data: {test_path}")
    print(f"Adapter: {adapter_path}")
    print(f"Output: {output_dir}")

    # Check paths
    if not test_path.exists():
        print(f"Error: Test data not found: {test_path}")
        return 1

    if not adapter_path.exists():
        print(f"Error: Adapter not found: {adapter_path}")
        return 1

    # Load test samples
    samples = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"Loaded {len(samples)} test samples")

    # Check GPU
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load model
    model, processor = load_model(str(adapter_path), use_4bit=not args.no_4bit)

    # Run inference
    results = []
    checkpoint_path = output_dir / f"predictions_checkpoint_{timestamp}.jsonl"

    for idx, sample in enumerate(tqdm(samples, desc="Running inference")):
        image_path = sample['images'][0]
        messages = sample['messages']

        # Extract question from user message
        user_msg = messages[1]
        question = None
        for content in user_msg['content']:
            if isinstance(content, dict) and content.get('type') == 'text':
                question = content['text']
                break

        # Extract ground truth from assistant message
        ground_truth = messages[2]['content']

        # Extract category
        category = extract_category_from_path(image_path)

        # Generate prediction
        try:
            prediction = generate_response(model, processor, image_path, question)
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

        # Checkpoint
        if (idx + 1) % args.checkpoint_interval == 0:
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
            print(f"\n  Checkpoint saved: {idx + 1}/{len(samples)} samples")

    # Save final predictions
    predictions_path = output_dir / f"predictions_{timestamp}.jsonl"
    with open(predictions_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print(f"\nInference complete!")
    print(f"Predictions saved to: {predictions_path}")
    print(f"\nTo download results:")
    print(f"  scp -P PORT root@IP:{predictions_path} ./")

    return 0


if __name__ == "__main__":
    exit(main())
EVAL_SCRIPT

echo "Evaluation script created at $WORKSPACE/run_inference.py"
echo ""

# 5. Final instructions
echo "[5/5] Setup complete!"
echo ""
echo "=============================================="
echo "  Next Steps"
echo "=============================================="
echo ""
echo "1. Upload test data (from local machine):"
echo "   scp -P PORT outputs/evaluation/test_subset.jsonl root@IP:$WORKSPACE/data/"
echo ""
echo "2. Upload LoRA adapter (from local machine):"
echo "   scp -P PORT -r models/qwen25vl7b_finetuned/final root@IP:$WORKSPACE/adapter/"
echo ""
echo "3. Upload image data (from local machine):"
echo "   scp -P PORT -r 'data/Fetal Ultrasound' root@IP:/workspace/data/"
echo ""
echo "4. Run inference:"
echo "   cd $WORKSPACE && python run_inference.py"
echo ""
echo "5. Download predictions (from local machine):"
echo "   scp -P PORT 'root@IP:$WORKSPACE/outputs/predictions_*.jsonl' outputs/evaluation/"
echo ""
echo "6. Score locally:"
echo "   python experiments/evaluation/evaluate_vlm.py --score-only --predictions outputs/evaluation/predictions_*.jsonl"
echo ""
echo "=============================================="
