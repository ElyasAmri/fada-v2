#!/usr/bin/env python3
"""
Vast.ai VLM Evaluation Orchestrator

Automates the evaluation workflow on Vast.ai:
1. Search for and rent GPU instance
2. Upload test data, adapter, and images
3. Setup environment and run inference
4. Download predictions
5. Score locally
6. Terminate instance

Usage:
    # Full automated workflow
    python eval_vastai.py

    # Just search for available GPUs
    python eval_vastai.py --search-only

    # Connect to existing instance
    python eval_vastai.py --instance-id 12345678

    # Download predictions from existing instance
    python eval_vastai.py --instance-id 12345678 --download-only

    # Score existing predictions locally
    python eval_vastai.py --score-only --predictions outputs/evaluation/predictions.jsonl
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re


# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "evaluation"
ADAPTER_PATH = PROJECT_ROOT / "models" / "qwen25vl7b_finetuned" / "final"
TEST_SUBSET = OUTPUTS_DIR / "test_subset.jsonl"
SETUP_SCRIPT = SCRIPT_DIR / "vastai_eval_setup.sh"


class VastaiEvaluator:
    """Manage Vast.ai instance for VLM evaluation"""

    def __init__(self, vastai_path: str = "vastai"):
        self.vastai = vastai_path
        self.instance_id: Optional[int] = None
        self.ssh_host: Optional[str] = None
        self.ssh_port: Optional[int] = None

    def search_gpus(
        self,
        gpu_name: str = "RTX_4090",
        min_vram: int = 20,
        max_price: float = 0.5,
        min_reliability: float = 0.95,
    ) -> List[Dict]:
        """Search for available GPU offers"""
        query_parts = [
            f"gpu_ram>={min_vram}",
            f"dph<{max_price}",
            f"reliability>{min_reliability}",
            "inet_down>100",
            "num_gpus=1",
        ]

        if gpu_name:
            query_parts.append(f"gpu_name={gpu_name.replace(' ', '_')}")

        query = " ".join(query_parts)
        cmd = [self.vastai, "search", "offers", query, "--raw"]
        print(f"Searching: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error searching offers: {result.stderr}")
            return []

        try:
            offers = json.loads(result.stdout)
            offers.sort(key=lambda x: x.get('dph_total', 999))
            return offers
        except json.JSONDecodeError:
            print(f"Failed to parse offers: {result.stdout[:500]}")
            return []

    def display_offers(self, offers: List[Dict], limit: int = 10) -> None:
        """Display available offers"""
        print(f"\n{'='*80}")
        print(f"{'ID':>10} | {'GPU':^20} | {'VRAM':^8} | {'$/hr':^8} | {'DL':^8} | {'Rel':^6}")
        print(f"{'='*80}")

        for offer in offers[:limit]:
            gpu = offer.get('gpu_name', 'Unknown')[:20]
            vram = offer.get('gpu_ram', 0)
            price = offer.get('dph_total', 0)
            dl_speed = offer.get('inet_down', 0)
            reliability = offer.get('reliability', 0)

            print(f"{offer['id']:>10} | {gpu:^20} | {vram:^8.0f} | ${price:^7.3f} | {dl_speed:^8.0f} | {reliability:^6.2f}")

        print(f"{'='*80}")
        print(f"Showing {min(limit, len(offers))} of {len(offers)} offers")

    def create_instance(
        self,
        offer_id: int,
        disk_gb: int = 80,
        image: str = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
    ) -> Optional[int]:
        """Create a Vast.ai instance"""
        cmd = [
            self.vastai, "create", "instance", str(offer_id),
            "--image", image,
            "--disk", str(disk_gb),
            "--ssh", "--direct",
        ]

        print(f"Creating instance: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error creating instance: {result.stderr}")
            return None

        try:
            match = re.search(r"'new_contract':\s*(\d+)", result.stdout)
            if match:
                return int(match.group(1))
            match = re.search(r"(\d+)", result.stdout)
            if match:
                return int(match.group(1))
        except Exception as e:
            print(f"Failed to parse instance ID: {e}")

        print(f"Create output: {result.stdout}")
        return None

    def wait_for_instance(self, instance_id: int, timeout: int = 600) -> bool:
        """Wait for instance to be ready"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            info = self.get_instance_info(instance_id)
            if info:
                status = info.get('actual_status', '')
                if status == 'running':
                    self.ssh_host = info.get('public_ipaddr', '')
                    ports = info.get('ports', {})
                    ssh_ports = ports.get('22/tcp', [])
                    if ssh_ports:
                        self.ssh_port = int(ssh_ports[0].get('HostPort', 22))
                    else:
                        self.ssh_port = info.get('ssh_port', 22)

                    if self.ssh_host and self.ssh_port:
                        return True

                elif status in ['error', 'terminated', 'failed']:
                    print(f"Instance {instance_id} failed: {status}")
                    return False

            print(f"Waiting for instance {instance_id}... (status: {status if info else 'unknown'})")
            time.sleep(15)

        print(f"Timeout waiting for instance {instance_id}")
        return False

    def get_instance_info(self, instance_id: int) -> Optional[Dict]:
        """Get instance info"""
        cmd = [self.vastai, "show", "instance", str(instance_id), "--raw"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            try:
                data = json.loads(result.stdout)
                if isinstance(data, list) and len(data) > 0:
                    return data[0]
                return data
            except json.JSONDecodeError:
                pass
        return None

    def run_ssh_command(
        self,
        command: str,
        timeout: int = 300,
        capture: bool = True,
    ) -> Tuple[int, str, str]:
        """Run command on remote instance via SSH"""
        ssh_cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=30",
            "-p", str(self.ssh_port),
            f"root@{self.ssh_host}",
            command,
        ]

        if capture:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode, result.stdout, result.stderr
        else:
            result = subprocess.run(ssh_cmd, timeout=timeout)
            return result.returncode, "", ""

    def upload_file(self, local_path: Path, remote_path: str) -> bool:
        """Upload file via SCP"""
        scp_cmd = [
            "scp",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-P", str(self.ssh_port),
            str(local_path),
            f"root@{self.ssh_host}:{remote_path}",
        ]

        result = subprocess.run(scp_cmd, capture_output=True, text=True)
        return result.returncode == 0

    def upload_directory(self, local_dir: Path, remote_dir: str) -> bool:
        """Upload directory via SCP"""
        scp_cmd = [
            "scp",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-P", str(self.ssh_port),
            "-r",
            str(local_dir),
            f"root@{self.ssh_host}:{remote_dir}",
        ]

        print(f"Uploading {local_dir} to {remote_dir}...")
        result = subprocess.run(scp_cmd)
        return result.returncode == 0

    def download_file(self, remote_path: str, local_path: Path) -> bool:
        """Download file via SCP"""
        local_path.parent.mkdir(parents=True, exist_ok=True)

        scp_cmd = [
            "scp",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-P", str(self.ssh_port),
            f"root@{self.ssh_host}:{remote_path}",
            str(local_path),
        ]

        print(f"Downloading {remote_path} to {local_path}...")
        result = subprocess.run(scp_cmd)
        return result.returncode == 0

    def setup_environment(self) -> bool:
        """Setup evaluation environment on remote"""
        print("\nSetting up evaluation environment...")

        setup_cmds = """
        pip install -q --upgrade pip
        pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
        pip install -q transformers>=4.45.0 accelerate bitsandbytes peft
        pip install -q pillow tqdm
        mkdir -p /workspace/fada-eval/data
        mkdir -p /workspace/fada-eval/adapter
        mkdir -p /workspace/fada-eval/outputs
        mkdir -p /workspace/data
        """

        ret, stdout, stderr = self.run_ssh_command(setup_cmds, timeout=600)
        if ret != 0:
            print(f"Setup failed: {stderr}")
            return False

        print("Environment setup complete.")
        return True

    def upload_evaluation_data(
        self,
        test_subset: Path,
        adapter_path: Path,
        images_dir: Path,
    ) -> bool:
        """Upload all evaluation data"""
        print("\nUploading evaluation data...")

        # Upload test subset
        print(f"  Uploading test subset: {test_subset.name}")
        if not self.upload_file(test_subset, "/workspace/fada-eval/data/test_subset.jsonl"):
            print("  Failed to upload test subset")
            return False

        # Upload adapter
        print(f"  Uploading adapter: {adapter_path}")
        if not self.upload_directory(adapter_path, "/workspace/fada-eval/"):
            print("  Failed to upload adapter")
            return False

        # Rename adapter directory on remote
        self.run_ssh_command("mv /workspace/fada-eval/final /workspace/fada-eval/adapter 2>/dev/null || true", timeout=30)

        # Upload images (this is the big one)
        print(f"  Uploading images: {images_dir}")
        if not self.upload_directory(images_dir, "/workspace/data/"):
            print("  Failed to upload images")
            return False

        print("Evaluation data uploaded.")
        return True

    def upload_inference_script(self) -> bool:
        """Upload the inference script (extracted from setup script)"""
        # Read setup script and extract the Python script
        if SETUP_SCRIPT.exists():
            with open(SETUP_SCRIPT, 'r') as f:
                content = f.read()

            # Extract Python script between markers
            start_marker = "cat > $WORKSPACE/run_inference.py << 'EVAL_SCRIPT'"
            end_marker = "EVAL_SCRIPT"

            start_idx = content.find(start_marker)
            if start_idx != -1:
                start_idx = content.find('\n', start_idx) + 1
                end_idx = content.find(end_marker, start_idx)
                if end_idx != -1:
                    script_content = content[start_idx:end_idx]

                    # Write to temp file
                    temp_script = SCRIPT_DIR / "temp_run_inference.py"
                    with open(temp_script, 'w') as f:
                        f.write(script_content)

                    # Upload
                    result = self.upload_file(temp_script, "/workspace/fada-eval/run_inference.py")

                    # Clean up
                    temp_script.unlink()
                    return result

        # Fallback: run setup script
        print("Uploading setup script and running it...")
        if self.upload_file(SETUP_SCRIPT, "/workspace/vastai_eval_setup.sh"):
            ret, _, _ = self.run_ssh_command("bash /workspace/vastai_eval_setup.sh", timeout=300)
            return ret == 0
        return False

    def run_inference(self, use_screen: bool = True) -> bool:
        """Start inference on remote"""
        print("\nStarting inference...")

        inference_cmd = """
        cd /workspace/fada-eval && python run_inference.py \
            --test-data data/test_subset.jsonl \
            --adapter adapter \
            --output outputs
        """

        if use_screen:
            screen_cmd = f'screen -dmS inference bash -c "{inference_cmd}; exec bash"'
            ret, _, stderr = self.run_ssh_command(screen_cmd, timeout=60)
            if ret == 0:
                print("Inference started in screen session 'inference'")
                print("  To attach: screen -r inference")
                print("  To detach: Ctrl+A, D")
                return True
            else:
                print(f"Failed to start screen session: {stderr}")
                return False
        else:
            ret, _, _ = self.run_ssh_command(inference_cmd, timeout=86400, capture=False)
            return ret == 0

    def check_inference_status(self) -> Dict:
        """Check inference status"""
        ret, stdout, _ = self.run_ssh_command("pgrep -f 'python.*run_inference.py'", timeout=30)
        is_running = ret == 0

        ret, predictions, _ = self.run_ssh_command(
            "ls -la /workspace/fada-eval/outputs/predictions_*.jsonl 2>/dev/null || echo 'No predictions yet'",
            timeout=30,
        )

        ret, progress, _ = self.run_ssh_command(
            "wc -l /workspace/fada-eval/outputs/predictions_*.jsonl 2>/dev/null || echo '0'",
            timeout=30,
        )

        return {
            "is_running": is_running,
            "predictions_files": predictions,
            "progress": progress,
        }

    def download_predictions(self, local_dir: Path) -> Optional[Path]:
        """Download prediction files"""
        local_dir.mkdir(parents=True, exist_ok=True)

        # Find prediction files
        ret, files, _ = self.run_ssh_command(
            "ls /workspace/fada-eval/outputs/predictions_*.jsonl 2>/dev/null | head -1",
            timeout=30,
        )

        if ret != 0 or not files.strip():
            print("No prediction files found")
            return None

        remote_path = files.strip()
        filename = Path(remote_path).name
        local_path = local_dir / filename

        if self.download_file(remote_path, local_path):
            return local_path
        return None

    def destroy_instance(self, instance_id: int) -> bool:
        """Terminate instance"""
        cmd = [self.vastai, "destroy", "instance", str(instance_id)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0


def score_predictions(predictions_path: Path, embedding_model: str = "all-mpnet-base-v2") -> Dict:
    """Score predictions locally using embedding similarity"""
    from embedding_scorer import EmbeddingScorer

    # Load predictions
    results = []
    with open(predictions_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    print(f"Loaded {len(results)} predictions from {predictions_path}")

    predictions = [r['prediction'] for r in results]
    ground_truths = [r['ground_truth'] for r in results]
    categories = [r['category'] for r in results]

    # Initialize scorer
    scorer = EmbeddingScorer(model_name=embedding_model, device="cpu")

    # Compute similarities
    similarities = scorer.compute_similarity(predictions, ground_truths)

    # Aggregate metrics
    aggregate = scorer.compute_aggregate_metrics(similarities)

    # Per-category breakdown
    category_metrics = scorer.compute_category_metrics(similarities, categories)

    return {
        "aggregate": aggregate,
        "per_category": category_metrics,
    }


def print_results_summary(scores: Dict):
    """Print formatted results summary"""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    agg = scores['aggregate']
    print(f"\nOverall Similarity Score: {agg['mean_similarity']:.4f}")
    print(f"Standard Deviation:       {agg['std_similarity']:.4f}")
    print(f"Median:                   {agg['median_similarity']:.4f}")
    print(f"Range:                    [{agg['min_similarity']:.4f}, {agg['max_similarity']:.4f}]")
    print(f"Number of Samples:        {agg['num_samples']}")

    print("\nPer-Category Breakdown:")
    print("-" * 60)
    for category, cat_scores in sorted(scores['per_category'].items()):
        mean = cat_scores['mean_similarity']
        std = cat_scores['std_similarity']
        n = cat_scores['num_samples']
        print(f"  {category:35s}: {mean:.4f} (+/- {std:.4f})  n={n}")


def main():
    parser = argparse.ArgumentParser(description="Vast.ai VLM Evaluation Orchestrator")

    # Instance selection
    parser.add_argument("--instance-id", type=int, help="Use existing instance ID")
    parser.add_argument("--gpu", type=str, default="RTX_4090", help="GPU type (RTX_4090, A100)")
    parser.add_argument("--min-vram", type=int, default=20, help="Minimum VRAM in GB")
    parser.add_argument("--max-price", type=float, default=0.5, help="Maximum price $/hr")

    # Paths
    parser.add_argument("--test-data", type=str, default=str(TEST_SUBSET))
    parser.add_argument("--adapter", type=str, default=str(ADAPTER_PATH))
    parser.add_argument("--images", type=str, default=str(DATA_DIR / "Fetal Ultrasound"))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUTS_DIR))

    # Workflow options
    parser.add_argument("--search-only", action="store_true", help="Only search GPUs")
    parser.add_argument("--upload-only", action="store_true", help="Only upload data")
    parser.add_argument("--status", action="store_true", help="Check inference status")
    parser.add_argument("--download-only", action="store_true", help="Only download predictions")
    parser.add_argument("--score-only", action="store_true", help="Only score locally")
    parser.add_argument("--predictions", type=str, help="Path to predictions file (for --score-only)")
    parser.add_argument("--destroy", action="store_true", help="Destroy instance after completion")

    # Other
    parser.add_argument("--vastai-path", type=str, default="vastai")
    parser.add_argument("--disk", type=int, default=80, help="Disk size in GB")
    parser.add_argument("--embedding-model", type=str, default="all-mpnet-base-v2")

    args = parser.parse_args()

    # Score-only mode (no vast.ai needed)
    if args.score_only:
        if not args.predictions:
            print("Error: --predictions required with --score-only")
            return 1

        predictions_path = Path(args.predictions)
        if not predictions_path.exists():
            print(f"Error: Predictions file not found: {predictions_path}")
            return 1

        print("Scoring predictions locally...")
        scores = score_predictions(predictions_path, args.embedding_model)
        print_results_summary(scores)

        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / f"evaluation_results_{predictions_path.stem}.json"
        with open(results_path, 'w') as f:
            json.dump(scores, f, indent=2)
        print(f"\nResults saved to: {results_path}")
        return 0

    evaluator = VastaiEvaluator(args.vastai_path)

    # Search only mode
    if args.search_only:
        print(f"Searching for {args.gpu} GPUs (min {args.min_vram}GB, max ${args.max_price}/hr)...")
        offers = evaluator.search_gpus(args.gpu, args.min_vram, args.max_price)
        evaluator.display_offers(offers)
        return 0

    # If instance ID provided, use it
    if args.instance_id:
        evaluator.instance_id = args.instance_id
        info = evaluator.get_instance_info(args.instance_id)
        if info:
            evaluator.ssh_host = info.get('public_ipaddr', '')
            ports = info.get('ports', {})
            ssh_ports = ports.get('22/tcp', [])
            if ssh_ports:
                evaluator.ssh_port = int(ssh_ports[0].get('HostPort', 22))
            print(f"Using instance {args.instance_id}: {evaluator.ssh_host}:{evaluator.ssh_port}")
        else:
            print(f"Could not find instance {args.instance_id}")
            return 1

    # Status check
    if args.status:
        if not evaluator.ssh_host:
            print("Error: Need --instance-id to check status")
            return 1
        status = evaluator.check_inference_status()
        print(f"\nInference running: {status['is_running']}")
        print(f"\nPrediction files:\n{status['predictions_files']}")
        print(f"\nProgress (lines):\n{status['progress']}")
        return 0

    # Download only
    if args.download_only:
        if not evaluator.ssh_host:
            print("Error: Need --instance-id to download")
            return 1
        output_dir = Path(args.output_dir)
        predictions_path = evaluator.download_predictions(output_dir)
        if predictions_path:
            print(f"\nPredictions downloaded to: {predictions_path}")
            print(f"\nTo score locally:")
            print(f"  python eval_vastai.py --score-only --predictions {predictions_path}")
        return 0

    # Full workflow
    try:
        # Step 1: Get or create instance
        if not evaluator.instance_id:
            print(f"\n[1/6] Searching for {args.gpu} GPUs...")
            offers = evaluator.search_gpus(args.gpu, args.min_vram, args.max_price)

            if not offers:
                print("No suitable offers found. Try adjusting --max-price or --gpu")
                return 1

            evaluator.display_offers(offers, limit=5)

            offer = offers[0]
            print(f"\nSelected offer {offer['id']}: {offer['gpu_name']}, ${offer['dph_total']:.3f}/hr")

            confirm = input("Create instance? [y/N]: ")
            if confirm.lower() != 'y':
                print("Aborted.")
                return 0

            print(f"\n[2/6] Creating instance...")
            evaluator.instance_id = evaluator.create_instance(offer['id'], disk_gb=args.disk)

            if not evaluator.instance_id:
                print("Failed to create instance")
                return 1

            print(f"Instance created: {evaluator.instance_id}")

            print(f"\n[3/6] Waiting for instance to be ready...")
            if not evaluator.wait_for_instance(evaluator.instance_id):
                print("Instance failed to start")
                return 1

            print(f"Instance ready: {evaluator.ssh_host}:{evaluator.ssh_port}")

        # Step 2: Setup environment
        print(f"\n[4/6] Setting up environment...")
        if not evaluator.setup_environment():
            print("Environment setup failed")
            return 1

        # Upload inference script
        print("Uploading inference script...")
        evaluator.upload_inference_script()

        # Step 3: Upload data
        print(f"\n[5/6] Uploading evaluation data...")
        test_data = Path(args.test_data)
        adapter_path = Path(args.adapter)
        images_path = Path(args.images)

        if not test_data.exists():
            print(f"Error: Test data not found: {test_data}")
            return 1
        if not adapter_path.exists():
            print(f"Error: Adapter not found: {adapter_path}")
            return 1
        if not images_path.exists():
            print(f"Error: Images not found: {images_path}")
            return 1

        if not evaluator.upload_evaluation_data(test_data, adapter_path, images_path):
            print("Data upload failed")
            return 1

        if args.upload_only:
            print("\nData uploaded. Connect and run manually:")
            print(f"  SSH: ssh -p {evaluator.ssh_port} root@{evaluator.ssh_host}")
            print(f"  Run: cd /workspace/fada-eval && python run_inference.py")
            return 0

        # Step 4: Run inference
        print(f"\n[6/6] Starting inference...")
        if not evaluator.run_inference():
            print("Failed to start inference")
            return 1

        print("\n" + "=" * 60)
        print("  Inference Started!")
        print("=" * 60)
        print(f"\nInstance ID: {evaluator.instance_id}")
        print(f"SSH: ssh -p {evaluator.ssh_port} root@{evaluator.ssh_host}")
        print(f"\nMonitor progress:")
        print(f"  python eval_vastai.py --instance-id {evaluator.instance_id} --status")
        print(f"\nDownload when complete:")
        print(f"  python eval_vastai.py --instance-id {evaluator.instance_id} --download-only")
        print(f"\nDestroy instance:")
        print(f"  vastai destroy instance {evaluator.instance_id}")

    except KeyboardInterrupt:
        print("\n\nAborted by user.")

    finally:
        if args.destroy and evaluator.instance_id:
            print(f"\nDestroying instance {evaluator.instance_id}...")
            evaluator.destroy_instance(evaluator.instance_id)

    return 0


if __name__ == "__main__":
    exit(main())
