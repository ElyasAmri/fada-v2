#!/usr/bin/env python3
"""
Unified Vast.ai CLI for VLM Testing and Fine-Tuning.

Usage:
    python -m experiments.vastai status                    # List all jobs
    python -m experiments.vastai test InternVL3-2B         # Quick test (20 samples)
    python -m experiments.vastai test InternVL3-2B --full  # Full evaluation (600 samples)
    python -m experiments.vastai batch-test A B C          # Test multiple models
    python -m experiments.vastai destroy job-abc123        # Destroy instance
    python -m experiments.vastai destroy --all             # Destroy all instances
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from .jobs import JobDatabase, JobStatus, JobType, Job, print_jobs_table
from .instance import VastInstance
from .presets import (
    PRESETS, MODEL_CONFIGS, resolve_model_id, get_preset_for_model,
    print_presets, print_model_list
)
from .transfer import DataTransfer, setup_environment, check_gpu, REMOTE_WORKSPACE


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "vastai"
TEST_SUBSET = PROJECT_ROOT / "outputs" / "evaluation" / "test_subset.jsonl"
IMAGES_DIR = DATA_DIR / "Fetal Ultrasound"
TEMPLATES_DIR = Path(__file__).parent / "templates"


class VastaiRunner:
    """Main orchestrator for Vast.ai operations."""

    def __init__(self):
        self.db = JobDatabase()
        self.instance = VastInstance()

    # =========================================================================
    # Status Commands
    # =========================================================================

    def cmd_status(self, job_id: str = None):
        """Show job status."""
        if job_id:
            job = self.db.get_job(job_id)
            if not job:
                print(f"Job not found: {job_id}")
                return 1

            print(f"\nJob: {job.job_id}")
            print(f"  Type: {job.job_type}")
            print(f"  Model: {job.model_id}")
            print(f"  Status: {job.status}")
            print(f"  Instance: {job.instance_id or 'None'}")
            if job.ssh_host:
                print(f"  SSH: ssh -p {job.ssh_port} root@{job.ssh_host}")
            print(f"  Created: {job.created_at}")
            if job.output_path:
                print(f"  Output: {job.output_path}")
            if job.error_message:
                print(f"  Error: {job.error_message}")

            # If instance is running, check remote status
            if job.instance_id and job.status == JobStatus.RUNNING.value:
                if self.instance.connect(job.instance_id):
                    if self.instance.is_process_running("run_inference.py"):
                        print(f"  Remote: Inference running")
                    else:
                        print(f"  Remote: No inference process found")
            return 0

        # List all jobs
        jobs = self.db.list_jobs(limit=20)
        print_jobs_table(jobs)

        # Show active instances
        active = self.db.get_active_jobs()
        if active:
            print(f"\nActive instances: {len(active)}")
            for job in active:
                if job.ssh_host:
                    print(f"  {job.job_id}: ssh -p {job.ssh_port} root@{job.ssh_host}")

        return 0

    # =========================================================================
    # Test Command
    # =========================================================================

    def cmd_test(
        self,
        model_id: str,
        samples: int = 20,
        full: bool = False,
        preset: str = None,
        dry_run: bool = False,
        auto_destroy: bool = False,
        yes: bool = False,
        skip_images: bool = False,
    ):
        """
        Test a VLM model on the evaluation dataset.

        Args:
            model_id: Model name, alias, or HuggingFace ID
            samples: Number of samples (default 20, ignored if --full)
            full: Run full 600-sample evaluation
            preset: GPU preset override
            dry_run: Show plan without executing
            auto_destroy: Destroy instance after completion
            yes: Skip confirmation prompts
        """
        # Resolve model ID
        model_id = resolve_model_id(model_id)

        # Get preset
        if preset:
            gpu_preset = PRESETS.get(preset)
            if not gpu_preset:
                print(f"Unknown preset: {preset}")
                print_presets()
                return 1
        else:
            gpu_preset = get_preset_for_model(model_id)

        # Sample count
        if full:
            samples = None  # All samples
            sample_desc = "all (600)"
        else:
            sample_desc = str(samples)

        print("\n" + "=" * 60)
        print("VLM Test Configuration")
        print("=" * 60)
        print(f"Model:    {model_id}")
        print(f"Samples:  {sample_desc}")
        print(f"Preset:   {gpu_preset.name} ({gpu_preset.description})")
        print(f"GPU:      {gpu_preset.gpu_name}")
        print(f"Max $/hr: ${gpu_preset.max_price:.2f}")

        if dry_run:
            print("\n[DRY RUN] Would execute the above configuration")
            return 0

        # Check test data exists
        if not TEST_SUBSET.exists():
            print(f"\nError: Test data not found: {TEST_SUBSET}")
            print("Run: python experiments/evaluation/create_test_subset.py")
            return 1

        # Check images exist
        if not IMAGES_DIR.exists():
            print(f"\nError: Images not found: {IMAGES_DIR}")
            return 1

        # Confirmation
        if not yes:
            confirm = input("\nProceed with test? [y/N]: ")
            if confirm.lower() != 'y':
                print("Aborted.")
                return 0

        # Create job record
        job = self.db.create_job(
            job_type=JobType.TEST,
            model_id=model_id,
            preset=gpu_preset.name,
            samples=samples,
            metadata={"full": full, "auto_destroy": auto_destroy}
        )
        print(f"\nJob created: {job.job_id}")

        try:
            # Step 1: Search for GPU
            print(f"\n[1/5] Searching for {gpu_preset.gpu_name}...")
            offers = self.instance.search_offers(
                gpu_name=gpu_preset.gpu_name,
                min_vram=gpu_preset.min_vram,
                max_price=gpu_preset.max_price,
            )

            if not offers:
                print("No suitable GPU offers found. Try:")
                print(f"  - Increase --max-price (current: ${gpu_preset.max_price})")
                print(f"  - Use a different preset")
                self.db.set_status(job.job_id, JobStatus.FAILED, "No GPU offers found")
                return 1

            self.instance.print_offers(offers, limit=5)
            offer = offers[0]
            print(f"\nSelected: {offer.gpu_name} @ ${offer.price_per_hour:.3f}/hr")

            # Step 2: Create instance
            print(f"\n[2/5] Creating instance...")
            if not self.instance.create(offer.offer_id, disk_gb=gpu_preset.disk_gb):
                self.db.set_status(job.job_id, JobStatus.FAILED, "Failed to create instance")
                return 1

            print(f"Instance created: {self.instance.instance_id}")
            self.db.set_status(job.job_id, JobStatus.CREATING)

            # Step 3: Wait for instance
            print(f"\n[3/5] Waiting for instance...")
            if not self.instance.wait_until_ready(timeout=600):
                self.db.set_status(job.job_id, JobStatus.FAILED, "Instance failed to start")
                return 1

            self.db.set_instance(
                job.job_id,
                self.instance.instance_id,
                self.instance.ssh_host,
                self.instance.ssh_port
            )
            print(f"Instance ready: {self.instance.ssh_command}")

            # Check GPU
            gpu_info = check_gpu(self.instance)
            if gpu_info:
                print(f"GPU: {gpu_info['name']}, VRAM: {gpu_info['memory_total']}")

            # Step 4: Setup environment and upload data
            print(f"\n[4/5] Setting up environment...")
            self.db.set_status(job.job_id, JobStatus.UPLOADING)

            if not setup_environment(self.instance):
                self.db.set_status(job.job_id, JobStatus.FAILED, "Environment setup failed")
                return 1

            transfer = DataTransfer(self.instance)
            if not transfer.setup_workspace():
                self.db.set_status(job.job_id, JobStatus.FAILED, "Workspace setup failed")
                return 1

            print("Uploading data...")
            if not transfer.upload_test_data(TEST_SUBSET):
                self.db.set_status(job.job_id, JobStatus.FAILED, "Failed to upload test data")
                return 1

            if skip_images:
                print("  Skipping image upload (--skip-images)")
            elif not transfer.upload_test_images(TEST_SUBSET):
                # Fall back to full upload if test images fail
                print("  Falling back to full image upload...")
                if not transfer.upload_images(IMAGES_DIR):
                    self.db.set_status(job.job_id, JobStatus.FAILED, "Failed to upload images")
                    return 1

            # Upload inference script
            inference_script = TEMPLATES_DIR / "run_inference.py"
            if not transfer.upload_script(inference_script):
                self.db.set_status(job.job_id, JobStatus.FAILED, "Failed to upload script")
                return 1

            # Step 5: Run inference
            print(f"\n[5/5] Starting inference...")
            self.db.set_status(job.job_id, JobStatus.RUNNING)

            samples_arg = f"--samples {samples}" if samples else ""
            inference_cmd = f"""
            cd {REMOTE_WORKSPACE} && python scripts/run_inference.py \
                --model {model_id} \
                --test-data data/test_subset.jsonl \
                --output outputs \
                {samples_arg}
            """

            # Run in foreground (wait for completion)
            print("Running inference (this may take a while)...")
            ret, stdout, stderr = self.instance.run_ssh(
                inference_cmd,
                timeout=7200,  # 2 hour timeout
                capture=True
            )

            if ret != 0:
                print(f"Inference failed: {stderr}")
                self.db.set_status(job.job_id, JobStatus.FAILED, f"Inference failed: {stderr[:200]}")
                return 1

            print("Inference complete!")

            # Step 6: Download results
            print("\nDownloading results...")
            self.db.set_status(job.job_id, JobStatus.DOWNLOADING)

            output_dir = OUTPUTS_DIR / job.job_id
            predictions_path = transfer.download_predictions(output_dir)

            if predictions_path:
                print(f"Predictions saved to: {predictions_path}")
                self.db.set_output(job.job_id, str(predictions_path))

                # Score locally
                print("\nScoring predictions...")
                self.db.set_status(job.job_id, JobStatus.SCORING)
                score = self._score_predictions(predictions_path)
                if score:
                    print(f"\nSimilarity Score: {score:.4f}")
            else:
                print("Warning: Could not download predictions")

            self.db.set_status(job.job_id, JobStatus.COMPLETE)
            print(f"\nJob complete: {job.job_id}")

        except KeyboardInterrupt:
            print("\n\nAborted by user.")
            self.db.set_status(job.job_id, JobStatus.CANCELLED, "User abort")

        except Exception as e:
            print(f"\nError: {e}")
            self.db.set_status(job.job_id, JobStatus.FAILED, str(e))
            return 1

        finally:
            if auto_destroy and self.instance.instance_id:
                print(f"\nDestroying instance {self.instance.instance_id}...")
                self.instance.destroy()

        return 0

    def _score_predictions(self, predictions_path: Path) -> float:
        """Score predictions locally using embedding similarity."""
        try:
            # Import scorer from evaluation module
            import sys
            eval_dir = PROJECT_ROOT / "experiments" / "evaluation"
            sys.path.insert(0, str(eval_dir))
            from embedding_scorer import EmbeddingScorer

            import json
            results = []
            with open(predictions_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))

            predictions = [r['prediction'] for r in results]
            ground_truths = [r['ground_truth'] for r in results]

            scorer = EmbeddingScorer(device="cpu")
            similarities = scorer.compute_similarity(predictions, ground_truths)
            metrics = scorer.compute_aggregate_metrics(similarities)

            return metrics['mean_similarity']

        except Exception as e:
            print(f"Scoring failed: {e}")
            return None

    # =========================================================================
    # Batch Test Command
    # =========================================================================

    def cmd_batch_test(
        self,
        models: list,
        samples: int = 20,
        auto_destroy: bool = True,
        yes: bool = False,
    ):
        """
        Test multiple models in sequence.

        Args:
            models: List of model names/IDs
            samples: Samples per model
            auto_destroy: Destroy instance after each model
            yes: Skip confirmations
        """
        print("\n" + "=" * 60)
        print("Batch Test Configuration")
        print("=" * 60)
        print(f"Models: {len(models)}")
        for m in models:
            resolved = resolve_model_id(m)
            preset = get_preset_for_model(resolved)
            print(f"  - {resolved} ({preset.name})")
        print(f"Samples per model: {samples}")
        print(f"Auto-destroy: {auto_destroy}")

        if not yes:
            confirm = input("\nProceed with batch test? [y/N]: ")
            if confirm.lower() != 'y':
                print("Aborted.")
                return 0

        results = []
        for i, model in enumerate(models, 1):
            print(f"\n{'='*60}")
            print(f"Model {i}/{len(models)}: {model}")
            print("=" * 60)

            ret = self.cmd_test(
                model_id=model,
                samples=samples,
                auto_destroy=auto_destroy,
                yes=True,
            )

            results.append({
                "model": resolve_model_id(model),
                "success": ret == 0,
            })

        # Summary
        print("\n" + "=" * 60)
        print("Batch Test Summary")
        print("=" * 60)
        for r in results:
            status = "OK" if r["success"] else "FAILED"
            print(f"  {r['model']}: {status}")

        return 0

    # =========================================================================
    # Destroy Command
    # =========================================================================

    def cmd_destroy(self, job_id: str = None, all_instances: bool = False):
        """Destroy instances."""
        if all_instances:
            active = self.db.get_active_jobs()
            if not active:
                print("No active instances to destroy.")
                return 0

            print(f"Destroying {len(active)} instances...")
            for job in active:
                if job.instance_id:
                    print(f"  Destroying {job.job_id} (instance {job.instance_id})...")
                    self.instance.destroy(job.instance_id)
                    self.db.set_status(job.job_id, JobStatus.CANCELLED, "User destroyed")
            return 0

        if not job_id:
            print("Specify job ID or --all")
            return 1

        job = self.db.get_job(job_id)
        if not job:
            print(f"Job not found: {job_id}")
            return 1

        if job.instance_id:
            print(f"Destroying instance {job.instance_id}...")
            self.instance.destroy(job.instance_id)
            self.db.set_status(job.job_id, JobStatus.CANCELLED, "User destroyed")
            print("Done.")
        else:
            print("No instance associated with this job.")

        return 0

    # =========================================================================
    # Download Command
    # =========================================================================

    def cmd_download(self, job_id: str):
        """Download results from a running job."""
        job = self.db.get_job(job_id)
        if not job:
            print(f"Job not found: {job_id}")
            return 1

        if not job.instance_id:
            print("No instance for this job.")
            return 1

        if not self.instance.connect(job.instance_id):
            print(f"Could not connect to instance {job.instance_id}")
            return 1

        print(f"Connected to {self.instance.ssh_command}")

        transfer = DataTransfer(self.instance)
        output_dir = OUTPUTS_DIR / job_id

        print("Downloading results...")
        downloaded = transfer.download_results(output_dir)

        if downloaded:
            print(f"\nDownloaded {len(downloaded)} files to {output_dir}")
            for f in downloaded:
                print(f"  - {f.name}")
        else:
            print("No results found to download.")

        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Unified Vast.ai CLI for VLM Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m experiments.vastai status
  python -m experiments.vastai test InternVL3-2B --samples 20
  python -m experiments.vastai test Qwen2.5-VL-3B --full --auto-destroy
  python -m experiments.vastai batch-test InternVL3-2B Qwen2.5-VL-3B SmolVLM2-2B
  python -m experiments.vastai destroy job-abc123
  python -m experiments.vastai models  # List supported models
  python -m experiments.vastai presets # List GPU presets
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show job status")
    status_parser.add_argument("job_id", nargs="?", help="Job ID (optional)")

    # Test command
    test_parser = subparsers.add_parser("test", help="Test a VLM model")
    test_parser.add_argument("model", help="Model name or HuggingFace ID")
    test_parser.add_argument("--samples", type=int, default=20, help="Number of samples")
    test_parser.add_argument("--full", action="store_true", help="Full 600-sample evaluation")
    test_parser.add_argument("--preset", help="GPU preset override")
    test_parser.add_argument("--dry-run", action="store_true", help="Show plan only")
    test_parser.add_argument("--auto-destroy", action="store_true", help="Destroy after completion")
    test_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmations")
    test_parser.add_argument("--skip-images", action="store_true", help="Skip image upload (if already on instance)")

    # Batch test command
    batch_parser = subparsers.add_parser("batch-test", help="Test multiple models")
    batch_parser.add_argument("models", nargs="+", help="Model names")
    batch_parser.add_argument("--samples", type=int, default=20, help="Samples per model")
    batch_parser.add_argument("--no-auto-destroy", action="store_true")
    batch_parser.add_argument("-y", "--yes", action="store_true")

    # Destroy command
    destroy_parser = subparsers.add_parser("destroy", help="Destroy instances")
    destroy_parser.add_argument("job_id", nargs="?", help="Job ID")
    destroy_parser.add_argument("--all", action="store_true", help="Destroy all instances")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download results")
    download_parser.add_argument("job_id", help="Job ID")

    # Models command
    subparsers.add_parser("models", help="List supported models")

    # Presets command
    subparsers.add_parser("presets", help="List GPU presets")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    runner = VastaiRunner()

    if args.command == "status":
        return runner.cmd_status(args.job_id)

    elif args.command == "test":
        return runner.cmd_test(
            model_id=args.model,
            samples=args.samples,
            full=args.full,
            preset=args.preset,
            dry_run=args.dry_run,
            auto_destroy=args.auto_destroy,
            yes=args.yes,
            skip_images=args.skip_images,
        )

    elif args.command == "batch-test":
        return runner.cmd_batch_test(
            models=args.models,
            samples=args.samples,
            auto_destroy=not args.no_auto_destroy,
            yes=args.yes,
        )

    elif args.command == "destroy":
        return runner.cmd_destroy(args.job_id, all_instances=args.all)

    elif args.command == "download":
        return runner.cmd_download(args.job_id)

    elif args.command == "models":
        print_model_list()
        return 0

    elif args.command == "presets":
        print_presets()
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
