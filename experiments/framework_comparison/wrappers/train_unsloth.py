"""
Unsloth training wrapper for framework comparison.

Uses Unsloth's FastModel for optimized LoRA fine-tuning with 4-bit quantization.
Outputs adapter in HF PEFT format.

Usage:
    python wrappers/train_unsloth.py \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --train-data /path/to/train.jsonl \
        --val-data /path/to/val.jsonl \
        --data-root /path/to/images \
        --output-dir /path/to/output \
        --config /path/to/config.json

    # Quick test
    python wrappers/train_unsloth.py --model Qwen/Qwen2.5-VL-7B-Instruct --test-run
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


class VLMDataset(Dataset):
    """Dataset for VLM fine-tuning from HF chat format JSONL."""

    def __init__(self, jsonl_path, processor, max_samples=None, max_length=4096,
                 data_root=None):
        self.processor = processor
        self.max_length = max_length
        self.data_root = Path(data_root) if data_root else None
        self._missing_count = 0
        self.samples = []

        self._response_template_ids = self.processor.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        self._end_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

        with open(jsonl_path) as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                self.samples.append(json.loads(line))

        print(f"Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        messages = sample["messages"]
        image_paths = sample.get("images", [])

        images = []
        for img_path in image_paths:
            if self.data_root and not os.path.isabs(img_path):
                full_path = (self.data_root / img_path).resolve()
                if not full_path.is_relative_to(self.data_root.resolve()):
                    raise ValueError(f"Path traversal detected: {img_path}")
                img_path = str(full_path)
            try:
                images.append(Image.open(img_path).convert("RGB"))
            except FileNotFoundError:
                self._missing_count += 1
                if self._missing_count > 10:
                    raise RuntimeError(f"Too many missing images ({self._missing_count}). Check --data-root.")
                images.append(Image.new("RGB", (224, 224), color="gray"))

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(
            text=text, images=images if images else None,
            return_tensors="pt", padding=False,
            truncation=True, max_length=self.max_length,
        )
        # Remove batch dim but keep image_grid_thw as 2D (num_images, 3)
        keep_2d = {"image_grid_thw", "video_grid_thw"}
        squeezed = {}
        for k, v in inputs.items():
            if k in keep_2d:
                # (1, num_images, 3) -> (num_images, 3)
                squeezed[k] = v.squeeze(0) if v.dim() == 3 else v
            else:
                squeezed[k] = v.squeeze(0)
        inputs = squeezed

        input_ids = inputs["input_ids"]
        labels = torch.full_like(input_ids, -100)

        if self._response_template_ids:
            template_len = len(self._response_template_ids)
            ids_list = input_ids.tolist()
            i = 0
            while i <= len(ids_list) - template_len:
                if ids_list[i:i + template_len] == self._response_template_ids:
                    j = i + template_len
                    while j < len(ids_list):
                        labels[j] = input_ids[j]
                        if ids_list[j] == self._end_token_id:
                            break
                        j += 1
                    i = j + 1
                else:
                    i += 1
        else:
            labels = input_ids.clone()
            labels[labels == self.processor.tokenizer.pad_token_id] = -100

        inputs["labels"] = labels
        return inputs


def collate_fn(batch):
    """Custom collator for variable-sized image tensors and dynamic padding.

    For Qwen VL models, pixel_values and image_grid_thw are variable-length
    and must be concatenated along dim 0 rather than stacked.
    Sequence tensors (input_ids, attention_mask, labels) are padded to the
    longest in the batch for efficiency.
    """
    concat_keys = {"pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"}
    # Pad sequence tensors to longest in batch
    pad_keys = {"input_ids", "attention_mask", "labels"}
    collated = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            if key in concat_keys:
                collated[key] = torch.cat(values, dim=0)
            elif key in pad_keys:
                max_len = max(v.shape[-1] for v in values)
                pad_value = -100 if key == "labels" else 0
                padded = []
                for v in values:
                    pad_size = max_len - v.shape[-1]
                    if pad_size > 0:
                        padded.append(torch.nn.functional.pad(v, (0, pad_size), value=pad_value))
                    else:
                        padded.append(v)
                collated[key] = torch.stack(padded)
            else:
                shapes = {v.shape for v in values}
                collated[key] = torch.stack(values) if len(shapes) == 1 else torch.cat(values, dim=0)
        else:
            collated[key] = values
    return collated


def main():
    parser = argparse.ArgumentParser(description="Unsloth LoRA fine-tuning wrapper")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--train-data", required=True, help="Training JSONL path")
    parser.add_argument("--val-data", default=None, help="Validation JSONL path")
    parser.add_argument("--data-root", required=True, help="Root directory for image paths")
    parser.add_argument("--output-dir", required=True, help="Output directory for adapter")
    parser.add_argument("--config", default=None, help="Path to config.json")
    parser.add_argument("--test-run", action="store_true", help="Quick test with 100 samples, 1 step")
    args = parser.parse_args()

    # Load config
    config_path = args.config or str(Path(__file__).parent.parent / "config.json")
    with open(config_path) as f:
        config = json.load(f)

    lora_cfg = config["lora"]
    train_cfg = config["training"]
    quant_cfg = config["quantization"]

    if args.test_run:
        max_samples = 100
        train_cfg = {**train_cfg, "num_train_epochs": 1, "logging_steps": 1, "save_strategy": "no",
                     "eval_strategy": "no"}
    else:
        max_samples = train_cfg.get("max_samples", None)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import unsloth
    from unsloth import FastModel

    print(f"Loading model with Unsloth: {args.model}")
    model, processor = FastModel.from_pretrained(
        args.model,
        max_seq_length=train_cfg["max_seq_length"],
        load_in_4bit=quant_cfg["load_in_4bit"],
        dtype=torch.bfloat16,
    )

    # Apply LoRA via Unsloth
    model = FastModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # Load datasets
    train_dataset = VLMDataset(
        args.train_data, processor, max_samples=max_samples,
        max_length=train_cfg["max_seq_length"], data_root=args.data_root,
    )
    eval_dataset = None
    if args.val_data and Path(args.val_data).exists():
        eval_dataset = VLMDataset(
            args.val_data, processor,
            max_samples=min(1000, max_samples) if max_samples else 1000,
            max_length=train_cfg["max_seq_length"], data_root=args.data_root,
        )

    # Enable TF32 for H100 (faster matmuls with minimal precision loss)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Training
    from transformers import TrainingArguments, Trainer

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        bf16=train_cfg["bf16"],
        tf32=True,
        optim=train_cfg["optim"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        save_steps=train_cfg.get("save_steps", 500),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        eval_strategy=train_cfg["eval_strategy"],
        eval_steps=train_cfg.get("eval_steps", 100),
        load_best_model_at_end=train_cfg["eval_strategy"] != "no",
        metric_for_best_model="eval_loss" if train_cfg["eval_strategy"] != "no" else None,
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
        report_to="none",
        seed=config["data"]["seed"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        data_collator=collate_fn,
    )

    print("Starting training...")
    start_time = time.time()
    gpu_mem_before = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    trainer.train()

    training_time = time.time() - start_time
    gpu_mem_peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

    # Extract final loss
    final_loss = None
    if trainer.state.log_history:
        for entry in reversed(trainer.state.log_history):
            if "loss" in entry:
                final_loss = entry["loss"]
                break

    # Save adapter in PEFT format
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    print(f"Adapter saved to {adapter_dir}")

    # Save training log history (loss curves, eval metrics)
    log_history = trainer.state.log_history if trainer.state.log_history else []
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(log_history, f, indent=2)
    print(f"Training log saved: {len(log_history)} entries")

    # Write run manifest
    manifest = {
        "model": args.model,
        "framework": "unsloth",
        "status": "complete",
        "lora_config": lora_cfg,
        "training_config": train_cfg,
        "training_time_seconds": round(training_time, 1),
        "final_loss": final_loss,
        "gpu_memory_peak_gb": round(gpu_mem_peak / 1e9, 2),
        "train_samples": len(train_dataset),
        "val_samples": len(eval_dataset) if eval_dataset else 0,
        "adapter_path": str(adapter_dir),
    }
    with open(output_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Training complete in {training_time:.0f}s | Loss: {final_loss} | GPU peak: {gpu_mem_peak / 1e9:.1f}GB")


if __name__ == "__main__":
    main()
