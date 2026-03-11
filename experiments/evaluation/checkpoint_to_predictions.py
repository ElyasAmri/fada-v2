"""
Convert test_api_vlm.py checkpoint file to predictions JSONL for scoring.

Usage:
    python -m experiments.evaluation.checkpoint_to_predictions \
        --checkpoint experiments/api_models/results/checkpoint_vllm_Qwen_Qwen3-VL-8B-Instruct.json \
        --output outputs/evaluation/predictions_qwen3vl8b.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config.questions import QUESTIONS


def convert(checkpoint_path: Path, output_path: Path) -> int:
    with open(checkpoint_path) as f:
        data = json.load(f)

    completed = data.get("completed_images", {})
    count = 0
    sample_id = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as out:
        for image_key, result in completed.items():
            category = result.get("category", image_key.split("/")[0])
            image_name = result.get("image", image_key.split("/")[-1])
            image_path = f"{category}/{image_name}"

            for q in result.get("questions", []):
                q_idx = q.get("question_idx", 0)
                question_text = QUESTIONS[q_idx] if q_idx < len(QUESTIONS) else q.get("question", "")
                response = q.get("response", "")

                pred = {
                    "sample_id": sample_id,
                    "image_path": image_path,
                    "category": category,
                    "question": question_text,
                    "prediction": response,
                }
                out.write(json.dumps(pred) + "\n")
                count += 1
                sample_id += 1

    print(f"Converted {len(completed)} images -> {count} predictions to {output_path}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Convert checkpoint to predictions JSONL")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    convert(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
