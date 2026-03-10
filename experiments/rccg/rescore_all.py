"""Batch rescore all checkpoints with the updated scoring pipeline."""
import subprocess
import sys
import glob
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PYTHON = os.path.join("venv", "Scripts", "python.exe")
CONVERT_SCRIPT = os.path.join("experiments", "evaluation", "checkpoint_to_predictions.py")
SCORE_SCRIPT = os.path.join("experiments", "evaluation", "score_against_gt.py")

def main():
    checkpoints = sorted(glob.glob(os.path.join(RESULTS_DIR, "checkpoint_*.json")))
    print(f"Found {len(checkpoints)} checkpoints")

    for cp in checkpoints:
        basename = os.path.basename(cp)
        # Extract model name from checkpoint filename
        # checkpoint_vllm_Qwen_Qwen3.5-2B.json -> Qwen_Qwen3.5-2B
        model_name = basename.replace("checkpoint_vllm_", "").replace("checkpoint_", "").replace(".json", "")

        pred_file = os.path.join(RESULTS_DIR, f"predictions_{model_name}.jsonl")
        score_file = os.path.join(RESULTS_DIR, f"scores_{model_name}.json")

        # Convert checkpoint to predictions if needed
        if not os.path.exists(pred_file):
            print(f"Converting {basename} -> predictions_{model_name}.jsonl")
            result = subprocess.run(
                [PYTHON, CONVERT_SCRIPT, "--checkpoint", cp, "--output", pred_file],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                print(f"  ERROR converting: {result.stderr[:200]}")
                continue

        # Always rescore with updated pipeline
        print(f"Scoring {model_name}...")
        result = subprocess.run(
            [PYTHON, SCORE_SCRIPT, "--predictions", pred_file, "--output", score_file],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  ERROR scoring: {result.stderr[:200]}")
            continue

        # Print result
        import json
        try:
            scores = json.load(open(score_file))
            overall = scores.get("overall", {})
            primary = overall.get("primary_score_mean", 0)
            embed = overall.get("embedding_similarity_mean", 0)
            print(f"  -> primary={primary:.4f}, embed_sim={embed:.4f}")
        except Exception as e:
            print(f"  ERROR reading scores: {e}")

    print("\nDone! All models rescored.")

if __name__ == "__main__":
    main()
