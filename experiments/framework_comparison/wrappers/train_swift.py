"""
ms-swift training wrapper for framework comparison.

Invokes `swift sft` CLI with appropriate arguments.
Uses HF chat format data natively. Outputs adapter in HF PEFT format.

Usage:
    python wrappers/train_swift.py \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --train-data /path/to/train.jsonl \
        --val-data /path/to/val.jsonl \
        --data-root /path/to/images \
        --output-dir /path/to/output \
        --config /path/to/config.json
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# ms-swift model type detection
MODEL_TYPES = {
    "qwen2.5-vl": "qwen2_5_vl",
    "qwen3-vl": "qwen3_vl",
    "qwen3.5": "qwen3_vl",
    "internvl3": "internvl3",
    "internvl2": "internvl2",
    "minicpm-v": "minicpm_v",
    "molmo": "molmo",
    "glm-4v": "glm4v",
    "phi-4": "phi4_mm",
}


def detect_model_type(model_id: str) -> str:
    """Detect ms-swift model type from model ID."""
    model_lower = model_id.lower()
    for key, mtype in MODEL_TYPES.items():
        if key in model_lower:
            return mtype
    return "auto"


def main():
    parser = argparse.ArgumentParser(description="ms-swift LoRA fine-tuning wrapper")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--train-data", required=True, help="Training JSONL path (HF chat format)")
    parser.add_argument("--val-data", default=None, help="Validation JSONL path")
    parser.add_argument("--data-root", required=True, help="Root directory for image paths")
    parser.add_argument("--output-dir", required=True, help="Output directory for adapter")
    parser.add_argument("--config", default=None, help="Path to config.json")
    parser.add_argument("--test-run", action="store_true", help="Quick test with 100 samples")
    args = parser.parse_args()

    config_path = args.config or str(Path(__file__).parent.parent / "config.json")
    with open(config_path) as f:
        config = json.load(f)

    lora_cfg = config["lora"]
    train_cfg = config["training"]
    quant_cfg = config["quantization"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_type = detect_model_type(args.model)

    # Build swift sft command
    cmd = [
        "swift", "sft",
        "--model", args.model,
        "--model_type", model_type,
        "--sft_type", "lora",
        "--dataset", str(Path(args.train_data).resolve()),
        "--dataset_format", "messages",
        "--images_dir", str(Path(args.data_root).resolve()),
        "--output_dir", str(output_dir / "adapter"),
        "--lora_rank", str(lora_cfg["r"]),
        "--lora_alpha", str(lora_cfg["lora_alpha"]),
        "--lora_dropout", str(lora_cfg["lora_dropout"]),
        "--lora_target_modules", lora_cfg["target_modules"],
        "--num_train_epochs", str(train_cfg["num_train_epochs"]),
        "--per_device_train_batch_size", str(train_cfg["per_device_train_batch_size"]),
        "--gradient_accumulation_steps", str(train_cfg["gradient_accumulation_steps"]),
        "--learning_rate", str(train_cfg["learning_rate"]),
        "--warmup_ratio", str(train_cfg["warmup_ratio"]),
        "--lr_scheduler_type", train_cfg["lr_scheduler_type"],
        "--bf16", str(train_cfg["bf16"]),
        "--optim", train_cfg["optim"],
        "--weight_decay", str(train_cfg["weight_decay"]),
        "--max_grad_norm", str(train_cfg["max_grad_norm"]),
        "--gradient_checkpointing", str(train_cfg["gradient_checkpointing"]),
        "--logging_steps", str(train_cfg["logging_steps"]),
        "--save_strategy", train_cfg["save_strategy"],
        "--max_length", str(train_cfg["max_seq_length"]),
        "--dataloader_num_workers", str(train_cfg["dataloader_num_workers"]),
        "--seed", str(config["data"]["seed"]),
    ]

    if quant_cfg["load_in_4bit"]:
        cmd.extend(["--quantization_bit", "4", "--bnb_4bit_comp_dtype", "bfloat16"])

    if args.val_data and Path(args.val_data).exists():
        cmd.extend(["--val_dataset", str(Path(args.val_data).resolve())])

    if args.test_run:
        cmd.extend(["--max_samples", "100"])

    print(f"ms-swift model type: {model_type}")
    print(f"Training {args.model}...")
    print(f"Command: {' '.join(cmd)}")

    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    result = subprocess.run(cmd, capture_output=False, text=True)

    training_time = time.time() - start_time
    gpu_mem_peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

    if result.returncode != 0:
        manifest = {
            "model": args.model,
            "framework": "swift",
            "status": "failed",
            "error": f"swift sft exited with code {result.returncode}",
            "training_time_seconds": round(training_time, 1),
        }
        with open(output_dir / "run_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        sys.exit(1)

    # Find the actual output directory (swift creates timestamped subdirs)
    adapter_dir = output_dir / "adapter"
    actual_adapter = None
    if adapter_dir.exists():
        # swift may create a subdirectory with the checkpoint
        subdirs = sorted(adapter_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if subdirs:
            actual_adapter = str(subdirs[0])
        elif (adapter_dir / "adapter_config.json").exists():
            actual_adapter = str(adapter_dir)

    # Read trainer state for final loss
    final_loss = None
    for state_path in adapter_dir.rglob("trainer_state.json"):
        with open(state_path) as f:
            state = json.load(f)
        for entry in reversed(state.get("log_history", [])):
            if "loss" in entry:
                final_loss = entry["loss"]
                break
        break

    manifest = {
        "model": args.model,
        "framework": "swift",
        "status": "complete",
        "lora_config": lora_cfg,
        "training_config": train_cfg,
        "training_time_seconds": round(training_time, 1),
        "final_loss": final_loss,
        "gpu_memory_peak_gb": round(gpu_mem_peak / 1e9, 2),
        "adapter_path": actual_adapter or str(adapter_dir),
    }
    with open(output_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Training complete in {training_time:.0f}s | Loss: {final_loss} | GPU peak: {gpu_mem_peak / 1e9:.1f}GB")


if __name__ == "__main__":
    main()
