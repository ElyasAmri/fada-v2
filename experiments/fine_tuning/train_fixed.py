"""
Fixed Qwen2.5-VL LoRA Fine-tuning Script with proper data collation
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

import torch
from torch.utils.data import Dataset
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Model configurations
MODEL_CONFIGS = {
    'qwen3-vl-2b': 'Qwen/Qwen3-VL-2B-Instruct',
    'qwen3-vl-4b': 'Qwen/Qwen3-VL-4B-Instruct',
    'qwen3-vl-8b': 'Qwen/Qwen3-VL-8B-Instruct',
    'qwen2.5-vl-3b': 'Qwen/Qwen2.5-VL-3B-Instruct',
    'qwen2.5-vl-7b': 'Qwen/Qwen2.5-VL-7B-Instruct',
    'qwen2-vl-2b': 'Qwen/Qwen2-VL-2B-Instruct',
    'qwen2-vl-7b': 'Qwen/Qwen2-VL-7B-Instruct',
}


class VLMDataset(Dataset):
    """Dataset for VLM fine-tuning - returns raw data for collator"""

    def __init__(self, jsonl_path: str, max_samples: Optional[int] = None):
        self.samples = []
        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                self.samples.append(json.loads(line))
        print(f'Loaded {len(self.samples)} samples from {jsonl_path}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class VLMDataCollator:
    """Custom data collator that handles variable-size images"""

    def __init__(self, processor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_texts = []
        batch_images = []

        for sample in features:
            messages = sample['messages']
            image_paths = sample.get('images', [])

            # Load images
            images = []
            for img_path in image_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    # Resize to consistent size to avoid tensor mismatch
                    img = img.resize((512, 512))
                    images.append(img)
                except Exception as e:
                    print(f'Warning: Could not load {img_path}: {e}')
                    images.append(Image.new('RGB', (512, 512), color='gray'))

            # Apply chat template
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

            batch_texts.append(text)
            batch_images.append(images[0] if images else None)

        # Process batch
        inputs = self.processor(
            text=batch_texts,
            images=batch_images,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        # Create labels
        labels = inputs['input_ids'].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs['labels'] = labels

        return inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen2.5-vl-7b', choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--train-data', type=str, required=True)
    parser.add_argument('--val-data', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--gradient-accumulation', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--max-train-samples', type=int, default=None)
    parser.add_argument('--max-val-samples', type=int, default=None)
    parser.add_argument('--lora-r', type=int, default=16)
    parser.add_argument('--test-run', action='store_true')
    args = parser.parse_args()

    if args.test_run:
        args.max_train_samples = 100
        args.max_val_samples = 20
        args.epochs = 1
        print('*** TEST RUN MODE ***')

    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'runs/{args.model}_{timestamp}'

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Output: {output_dir}')
    print(f'CUDA: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    # Load model
    model_id = MODEL_CONFIGS[args.model]
    print(f'Loading model: {model_id}')

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        bias='none',
        task_type='CAUSAL_LM',
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)')

    # Load data
    print('Loading datasets...')
    train_dataset = VLMDataset(args.train_data, args.max_train_samples)
    val_dataset = VLMDataset(args.val_data, args.max_val_samples) if args.val_data else None

    # Create collator
    data_collator = VLMDataCollator(processor)

    # Training args
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        save_strategy='steps',
        save_steps=1000,
        eval_strategy='steps' if val_dataset else 'no',
        eval_steps=1000,
        gradient_checkpointing=True,
        optim='adamw_8bit',
        remove_unused_columns=False,
        dataloader_num_workers=4,
        report_to='none',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Save config
    config = {
        'model': args.model,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset) if val_dataset else 0,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'gradient_accumulation': args.gradient_accumulation,
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print('Starting training...')
    trainer.train()

    print('Saving model...')
    trainer.save_model(str(output_dir / 'final'))
    processor.save_pretrained(str(output_dir / 'final'))

    print(f'Done! Model saved to: {output_dir / "final"}')


if __name__ == '__main__':
    main()
