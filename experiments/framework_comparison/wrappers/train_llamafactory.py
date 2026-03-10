"""
LLaMA Factory training wrapper for framework comparison.

Generates a LLaMA Factory YAML config and invokes `llamafactory-cli train`.
Uses ShareGPT format data. Outputs adapter in HF PEFT format.

Usage:
    python wrappers/train_llamafactory.py \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --train-data /path/to/train_sharegpt.jsonl \
        --val-data /path/to/val_sharegpt.jsonl \
        --data-root /path/to/images \
        --output-dir /path/to/output \
        --config /path/to/config.json
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Map model IDs to LLaMA Factory model templates
MODEL_TEMPLATES = {
    "qwen2.5": "qwen2_vl",
    "qwen3": "qwen2_vl",
    "internvl": "internvl2",
    "minicpm": "minicpm_v",
    "mistral": "mistral",
    "molmo": "default",
    "glm": "glm4v",
    "phi": "phi4_mm",
}


def detect_template(model_id: str) -> str:
    """Detect LLaMA Factory template from model ID."""
    model_lower = model_id.lower()
    for key, template in MODEL_TEMPLATES.items():
        if key in model_lower:
            return template
    return "default"


def main():
    parser = argparse.ArgumentParser(description="LLaMA Factory LoRA fine-tuning wrapper")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--train-data", required=True, help="Training JSONL path (ShareGPT format)")
    parser.add_argument("--val-data", default=None, help="Validation JSONL path (ShareGPT format)")
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

    template = detect_template(args.model)

    # Register dataset in a temporary dataset_info.json
    dataset_info = {
        "fada_train": {
            "file_name": str(Path(args.train_data).resolve()),
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "images": "images"},
            "tags": {"role_tag": "from", "content_tag": "value",
                     "user_tag": "human", "assistant_tag": "gpt", "system_tag": "system"},
        }
    }
    if args.val_data and Path(args.val_data).exists():
        dataset_info["fada_val"] = {
            "file_name": str(Path(args.val_data).resolve()),
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "images": "images"},
            "tags": {"role_tag": "from", "content_tag": "value",
                     "user_tag": "human", "assistant_tag": "gpt", "system_tag": "system"},
        }

    dataset_info_path = output_dir / "dataset_info.json"
    with open(dataset_info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)

    # Build LLaMA Factory training config
    lf_config = {
        "stage": "sft",
        "model_name_or_path": args.model,
        "template": template,
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": lora_cfg["r"],
        "lora_alpha": lora_cfg["lora_alpha"],
        "lora_dropout": lora_cfg["lora_dropout"],
        "lora_target": lora_cfg["target_modules"],
        "dataset": "fada_train",
        "dataset_dir": str(output_dir),
        "image_dir": str(Path(args.data_root).resolve()),
        "cutoff_len": train_cfg["max_seq_length"],
        "output_dir": str(output_dir / "adapter"),
        "overwrite_output_dir": True,
        "per_device_train_batch_size": train_cfg["per_device_train_batch_size"],
        "gradient_accumulation_steps": train_cfg["gradient_accumulation_steps"],
        "learning_rate": train_cfg["learning_rate"],
        "num_train_epochs": train_cfg["num_train_epochs"],
        "lr_scheduler_type": train_cfg["lr_scheduler_type"],
        "warmup_ratio": train_cfg["warmup_ratio"],
        "bf16": train_cfg["bf16"],
        "optim": train_cfg["optim"],
        "weight_decay": train_cfg["weight_decay"],
        "max_grad_norm": train_cfg["max_grad_norm"],
        "gradient_checkpointing": train_cfg["gradient_checkpointing"],
        "logging_steps": train_cfg["logging_steps"],
        "save_strategy": train_cfg["save_strategy"],
        "dataloader_num_workers": train_cfg["dataloader_num_workers"],
        "report_to": "none",
        "seed": config["data"]["seed"],
        # Quantization
        "quantization_bit": 4 if quant_cfg["load_in_4bit"] else None,
        "quantization_method": "bitsandbytes",
    }

    if args.val_data and Path(args.val_data).exists():
        lf_config["eval_dataset"] = "fada_val"
        lf_config["eval_strategy"] = train_cfg["eval_strategy"]

    if args.test_run:
        lf_config["max_samples"] = 100
        lf_config["num_train_epochs"] = 1
        lf_config["logging_steps"] = 1

    # Write config YAML
    lf_config_path = output_dir / "llamafactory_config.yaml"
    with open(lf_config_path, "w") as f:
        yaml.dump(lf_config, f, default_flow_style=False)

    print(f"LLaMA Factory config written to {lf_config_path}")
    print(f"Template: {template}")
    print(f"Training {args.model}...")

    # Run training
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    cmd = ["llamafactory-cli", "train", str(lf_config_path)]
    result = subprocess.run(cmd, capture_output=False, text=True)

    training_time = time.time() - start_time
    gpu_mem_peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

    if result.returncode != 0:
        manifest = {
            "model": args.model,
            "framework": "llamafactory",
            "status": "failed",
            "error": f"llamafactory-cli exited with code {result.returncode}",
            "training_time_seconds": round(training_time, 1),
        }
        with open(output_dir / "run_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        sys.exit(1)

    # Read trainer_state.json for final loss
    final_loss = None
    trainer_state_path = output_dir / "adapter" / "trainer_state.json"
    if trainer_state_path.exists():
        with open(trainer_state_path) as f:
            state = json.load(f)
        for entry in reversed(state.get("log_history", [])):
            if "loss" in entry:
                final_loss = entry["loss"]
                break

    # Write manifest
    manifest = {
        "model": args.model,
        "framework": "llamafactory",
        "status": "complete",
        "lora_config": lora_cfg,
        "training_config": train_cfg,
        "training_time_seconds": round(training_time, 1),
        "final_loss": final_loss,
        "gpu_memory_peak_gb": round(gpu_mem_peak / 1e9, 2),
        "adapter_path": str(output_dir / "adapter"),
    }
    with open(output_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Training complete in {training_time:.0f}s | Loss: {final_loss} | GPU peak: {gpu_mem_peak / 1e9:.1f}GB")


if __name__ == "__main__":
    main()
