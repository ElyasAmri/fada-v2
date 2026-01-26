"""
Benchmark configuration for VLM fine-tuning experiments.

Each model will be trained on a separate vast.ai instance.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for a single model benchmark."""
    name: str  # Short name for logging
    model_id: str  # HuggingFace model ID
    min_vram: int  # Minimum VRAM in GB
    batch_size: int = 1
    gradient_accumulation: int = 8
    use_4bit: bool = True
    epochs: int = 1  # For benchmarking, 1 epoch is enough
    max_train_samples: Optional[int] = None  # None = use all
    max_val_samples: Optional[int] = None


# Models to benchmark - ordered by size
BENCHMARK_MODELS = [
    # Qwen2-VL series (oldest, but has 2B)
    ModelConfig(
        name="qwen2-vl-2b",
        model_id="Qwen/Qwen2-VL-2B-Instruct",
        min_vram=12,
        batch_size=2,
    ),

    # Qwen2.5-VL series (Jan 2025)
    ModelConfig(
        name="qwen2.5-vl-3b",
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        min_vram=16,
        batch_size=2,
    ),

    # Qwen2-VL 7B
    ModelConfig(
        name="qwen2-vl-7b",
        model_id="Qwen/Qwen2-VL-7B-Instruct",
        min_vram=24,
        batch_size=1,
        gradient_accumulation=16,
    ),

    # Qwen2.5-VL 7B
    ModelConfig(
        name="qwen2.5-vl-7b",
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        min_vram=24,
        batch_size=1,
        gradient_accumulation=16,
    ),
]

# Quick benchmark subset (for testing the pipeline)
QUICK_BENCHMARK_MODELS = [
    ModelConfig(
        name="qwen2-vl-2b",
        model_id="Qwen/Qwen2-VL-2B-Instruct",
        min_vram=12,
        batch_size=2,
        max_train_samples=100,
        max_val_samples=20,
    ),
    ModelConfig(
        name="qwen2.5-vl-3b",
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        min_vram=16,
        batch_size=2,
        max_train_samples=100,
        max_val_samples=20,
    ),
]

# GPU presets for different model sizes
GPU_PRESETS = {
    "small": {  # 2B-3B models
        "gpu_name": "RTX_4090",
        "min_vram": 16,
        "max_price": 0.40,
    },
    "medium": {  # 7B models
        "gpu_name": "RTX_4090",
        "min_vram": 24,
        "max_price": 0.50,
    },
    "large": {  # 8B+ models
        "gpu_name": "A100_PCIE",
        "min_vram": 40,
        "max_price": 1.00,
    },
}


def get_gpu_preset(model_config: ModelConfig) -> dict:
    """Get appropriate GPU preset for a model."""
    if model_config.min_vram <= 16:
        return GPU_PRESETS["small"]
    elif model_config.min_vram <= 24:
        return GPU_PRESETS["medium"]
    else:
        return GPU_PRESETS["large"]
