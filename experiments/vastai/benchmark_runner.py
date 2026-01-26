"""
Multi-instance VLM benchmarking runner for vast.ai.

Usage:
    # Run all benchmarks (one instance per model)
    python experiments/vastai/benchmark_runner.py

    # Quick test with 2 models
    python experiments/vastai/benchmark_runner.py --quick

    # Specific models only
    python experiments/vastai/benchmark_runner.py --models qwen2-vl-2b qwen2.5-vl-3b
"""

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from benchmark_config import (
    BENCHMARK_MODELS,
    QUICK_BENCHMARK_MODELS,
    ModelConfig,
    get_gpu_preset,
)
from instance import VastInstance, GPUOffer


# HuggingFace dataset info
HF_DATASET = "elyasamri/fetal-ultrasound-vlm"
HF_TRAIN_FILE = "excel_train.jsonl"
HF_VAL_FILE = "excel_val.jsonl"
HF_IMAGES_ARCHIVE = "fetal_ultrasound_images.tar.gz"

# Docker image
DOCKER_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"


def create_setup_script(model_config: ModelConfig, hf_token: str) -> str:
    """Generate the setup and training script for an instance."""

    script = f'''#!/bin/bash
set -e

echo "=== Setting up benchmark for {model_config.name} ==="
echo "Started at: $(date)"

# Set HF token
export HF_TOKEN="{hf_token}"

# Install dependencies
echo "Installing dependencies..."
pip install -q transformers>=4.49.0 accelerate>=1.3.0 peft bitsandbytes datasets mlflow pillow qwen-vl-utils huggingface_hub

# Login to HF
python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"

# Create workspace
mkdir -p /workspace/data /workspace/output
cd /workspace

# Clone repository
echo "Cloning repository..."
git clone https://github.com/ElyasAmri/fada-v2.git
cd fada-v2

# Download dataset from HuggingFace
echo "Downloading dataset from HuggingFace..."
python -c "
from huggingface_hub import hf_hub_download
import tarfile
import os

# Download JSONL files
print('Downloading training files...')
train_path = hf_hub_download(
    repo_id='{HF_DATASET}',
    filename='{HF_TRAIN_FILE}',
    repo_type='dataset',
    local_dir='/workspace/data'
)
val_path = hf_hub_download(
    repo_id='{HF_DATASET}',
    filename='{HF_VAL_FILE}',
    repo_type='dataset',
    local_dir='/workspace/data'
)

# Download and extract images
print('Downloading image archive...')
archive_path = hf_hub_download(
    repo_id='{HF_DATASET}',
    filename='{HF_IMAGES_ARCHIVE}',
    repo_type='dataset',
    local_dir='/workspace/data'
)

print('Extracting images...')
with tarfile.open(archive_path, 'r:gz') as tar:
    tar.extractall('/workspace/data/images')

print('Dataset ready!')
"

# Fix paths in JSONL (images/ -> /workspace/data/images/)
echo "Fixing image paths..."
sed -i 's|"images/|"/workspace/data/images/|g' /workspace/data/{HF_TRAIN_FILE}
sed -i 's|"images/|"/workspace/data/images/|g' /workspace/data/{HF_VAL_FILE}

# Verify setup
echo "Verifying setup..."
ls -la /workspace/data/
head -1 /workspace/data/{HF_TRAIN_FILE} | grep -o '"images": \\[[^]]*\\]' | head -c 100

# Run training
echo "Starting training for {model_config.name}..."
cd /workspace/fada-v2

python experiments/fine_tuning/train_qwen3vl_lora.py \\
    --model {model_config.name} \\
    --train-data /workspace/data/{HF_TRAIN_FILE} \\
    --val-data /workspace/data/{HF_VAL_FILE} \\
    --epochs {model_config.epochs} \\
    --batch-size {model_config.batch_size} \\
    --gradient-accumulation {model_config.gradient_accumulation} \\
    {"--max-train-samples " + str(model_config.max_train_samples) if model_config.max_train_samples else ""} \\
    {"--max-val-samples " + str(model_config.max_val_samples) if model_config.max_val_samples else ""} \\
    {"--no-4bit" if not model_config.use_4bit else ""} \\
    --output-dir /workspace/output/{model_config.name}

echo "Training complete!"
echo "Finished at: $(date)"

# Save results summary
python -c "
import json
import os
from pathlib import Path

output_dir = Path('/workspace/output/{model_config.name}')
config_file = list(output_dir.glob('*/config.json'))
if config_file:
    with open(config_file[0]) as f:
        config = json.load(f)

    # Find final checkpoint
    final_dir = output_dir / 'final' if (output_dir / 'final').exists() else list(output_dir.glob('*/final'))[0]

    result = {{
        'model': '{model_config.name}',
        'model_id': '{model_config.model_id}',
        'config': config,
        'output_dir': str(final_dir),
        'adapter_size_mb': sum(f.stat().st_size for f in final_dir.glob('*.safetensors')) / 1e6,
    }}

    with open('/workspace/output/{model_config.name}_result.json', 'w') as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
"
'''
    return script


class BenchmarkRunner:
    """Manages multiple vast.ai instances for parallel benchmarking."""

    def __init__(self, hf_token: str):
        self.hf_token = hf_token
        self.instances: Dict[str, VastInstance] = {}
        self.results: Dict[str, dict] = {}
        self.output_dir = Path("experiments/vastai/benchmark_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def provision_instance(self, model_config: ModelConfig) -> Optional[VastInstance]:
        """Provision a vast.ai instance for a model."""
        gpu_preset = get_gpu_preset(model_config)

        print(f"\n[{model_config.name}] Searching for {gpu_preset['gpu_name']} instance...")

        instance = VastInstance()

        # Search for offers
        offers = instance.search_offers(
            gpu_name=gpu_preset["gpu_name"],
            min_vram=gpu_preset["min_vram"],
            max_price=gpu_preset["max_price"],
            min_reliability=0.99,
            min_download=500,
        )

        if not offers:
            print(f"[{model_config.name}] No suitable instances found!")
            return None

        # Select best offer (first one, sorted by price)
        offer = offers[0]
        print(f"[{model_config.name}] Selected: {offer.gpu_name} @ ${offer.price_per_hour:.3f}/hr")

        # Create instance
        instance_id = instance.create_instance(
            offer_id=offer.offer_id,
            image=DOCKER_IMAGE,
            disk_gb=100,  # Enough for dataset + model
        )

        if not instance_id:
            print(f"[{model_config.name}] Failed to create instance!")
            return None

        print(f"[{model_config.name}] Created instance {instance_id}")

        # Wait for instance to be ready
        if not instance.wait_for_ready(timeout=600):
            print(f"[{model_config.name}] Instance failed to start!")
            return None

        # Validate SSH
        validation = instance.validate_instance(max_retries=5, retry_delay=20)
        if not validation.get("valid", False):
            print(f"[{model_config.name}] SSH validation failed: {validation.get('errors', [])}")
            return None

        print(f"[{model_config.name}] Instance ready at {instance.ssh_host}:{instance.ssh_port}")

        return instance

    def run_benchmark(self, model_config: ModelConfig, instance: VastInstance) -> dict:
        """Run benchmark on a provisioned instance."""
        print(f"\n[{model_config.name}] Starting benchmark...")

        # Generate and upload setup script
        script = create_setup_script(model_config, self.hf_token)

        # Write script to instance
        ret, _, stderr = instance.run_ssh(
            f"cat > /workspace/setup.sh << 'SCRIPT_EOF'\n{script}\nSCRIPT_EOF",
            timeout=60
        )

        if ret != 0:
            return {"error": f"Failed to upload script: {stderr}"}

        # Make executable and run
        ret, stdout, stderr = instance.run_ssh(
            "chmod +x /workspace/setup.sh && /workspace/setup.sh 2>&1",
            timeout=7200  # 2 hour timeout for full training
        )

        # Save full output
        output_file = self.output_dir / f"{model_config.name}_output.txt"
        with open(output_file, "w") as f:
            f.write(stdout or "")
            if stderr:
                f.write(f"\n\nSTDERR:\n{stderr}")

        if ret != 0:
            return {
                "error": f"Training failed with exit code {ret}",
                "output_file": str(output_file),
            }

        # Retrieve results
        ret, result_json, _ = instance.run_ssh(
            f"cat /workspace/output/{model_config.name}_result.json 2>/dev/null || echo '{{}}'",
            timeout=30
        )

        try:
            result = json.loads(result_json)
            result["output_file"] = str(output_file)
            result["success"] = True
            return result
        except json.JSONDecodeError:
            return {
                "error": "Failed to parse results",
                "output_file": str(output_file),
            }

    def run_all_benchmarks(
        self,
        models: List[ModelConfig],
        parallel: bool = True,
        max_workers: int = 4,
    ) -> Dict[str, dict]:
        """Run benchmarks for all models."""

        print(f"\n{'='*60}")
        print(f"Starting benchmarks for {len(models)} models")
        print(f"Parallel: {parallel}, Max workers: {max_workers}")
        print(f"{'='*60}\n")

        start_time = time.time()

        if parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}

                # Submit all provisioning tasks
                for model_config in models:
                    future = executor.submit(self._run_single_benchmark, model_config)
                    futures[future] = model_config.name

                # Collect results as they complete
                for future in as_completed(futures):
                    model_name = futures[future]
                    try:
                        result = future.result()
                        self.results[model_name] = result
                        print(f"\n[{model_name}] Completed: {result.get('success', False)}")
                    except Exception as e:
                        self.results[model_name] = {"error": str(e)}
                        print(f"\n[{model_name}] Failed with exception: {e}")
        else:
            for model_config in models:
                result = self._run_single_benchmark(model_config)
                self.results[model_config.name] = result

        total_time = time.time() - start_time

        # Save summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_time_seconds": total_time,
            "models": [asdict(m) for m in models],
            "results": self.results,
        }

        summary_file = self.output_dir / f"benchmark_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Benchmarks complete! Total time: {total_time/60:.1f} minutes")
        print(f"Results saved to: {summary_file}")
        print(f"{'='*60}\n")

        return self.results

    def _run_single_benchmark(self, model_config: ModelConfig) -> dict:
        """Run a single benchmark (provision + train + cleanup)."""
        instance = None
        try:
            # Provision
            instance = self.provision_instance(model_config)
            if not instance:
                return {"error": "Failed to provision instance"}

            self.instances[model_config.name] = instance

            # Run benchmark
            result = self.run_benchmark(model_config, instance)

            return result

        except Exception as e:
            return {"error": str(e)}

        finally:
            # Stop instance (don't destroy, user might want to inspect)
            if instance and instance.instance_id:
                print(f"[{model_config.name}] Stopping instance {instance.instance_id}...")
                instance.stop_instance()

    def cleanup_all(self, destroy: bool = False):
        """Stop or destroy all instances."""
        for name, instance in self.instances.items():
            if instance.instance_id:
                if destroy:
                    print(f"Destroying instance for {name}...")
                    instance.destroy_instance()
                else:
                    print(f"Stopping instance for {name}...")
                    instance.stop_instance()


def main():
    parser = argparse.ArgumentParser(description="Run VLM benchmarks on vast.ai")
    parser.add_argument(
        "--quick", action="store_true",
        help="Run quick benchmark with subset of data"
    )
    parser.add_argument(
        "--models", nargs="+",
        help="Specific models to benchmark (e.g., qwen2-vl-2b qwen2.5-vl-3b)"
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Run benchmarks sequentially instead of in parallel"
    )
    parser.add_argument(
        "--max-workers", type=int, default=4,
        help="Maximum parallel instances"
    )
    parser.add_argument(
        "--hf-token", type=str,
        help="HuggingFace token (or set HF_TOKEN env var)"
    )

    args = parser.parse_args()

    # Get HF token
    import os
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        # Try to get from huggingface_hub
        try:
            from huggingface_hub import HfApi
            hf_token = HfApi().token
        except:
            pass

    if not hf_token:
        print("Error: HuggingFace token required. Set HF_TOKEN env var or use --hf-token")
        return

    # Select models
    if args.quick:
        models = QUICK_BENCHMARK_MODELS
    elif args.models:
        models = [m for m in BENCHMARK_MODELS if m.name in args.models]
        if not models:
            print(f"No matching models found. Available: {[m.name for m in BENCHMARK_MODELS]}")
            return
    else:
        models = BENCHMARK_MODELS

    print(f"Models to benchmark: {[m.name for m in models]}")

    # Run benchmarks
    runner = BenchmarkRunner(hf_token)

    try:
        results = runner.run_all_benchmarks(
            models=models,
            parallel=not args.sequential,
            max_workers=args.max_workers,
        )

        # Print summary
        print("\n" + "="*60)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*60)

        for model_name, result in results.items():
            status = "SUCCESS" if result.get("success") else "FAILED"
            error = result.get("error", "")
            print(f"{model_name}: {status} {error}")

    except KeyboardInterrupt:
        print("\nInterrupted! Cleaning up...")
        runner.cleanup_all(destroy=False)


if __name__ == "__main__":
    main()
