"""
Full-Scale VQA Training Automation Script

This script trains all available categories with their complete datasets
and multiple epochs for production-ready models.

Usage:
    python train_all_full_scale.py [--epochs 3] [--categories all]

Examples:
    # Train all categories with 3 epochs
    python train_all_full_scale.py

    # Train specific categories
    python train_all_full_scale.py --categories abdomen,femur

    # Train with 5 epochs
    python train_all_full_scale.py --epochs 5
"""

import subprocess
import sys
from pathlib import Path
import argparse
import time
import json

# Category configurations with dataset sizes
CATEGORIES = {
    "Non_standard_NT": {
        "notebook": "notebooks/train_blip2_1epoch.ipynb",
        "num_images": 487,
        "output": "notebooks/train_blip2_non_standard_nt_full.ipynb"
    },
    "Abdomen": {
        "notebook": "notebooks/train_blip2_abdomen.ipynb",
        "num_images": 2424,
        "output": "notebooks/train_blip2_abdomen_full.ipynb"
    },
    "Femur": {
        "notebook": "notebooks/train_blip2_femur.ipynb",
        "num_images": 1165,
        "output": "notebooks/train_blip2_femur_full.ipynb"
    },
    "Thorax": {
        "notebook": "notebooks/train_blip2_thorax.ipynb",
        "num_images": 1793,
        "output": "notebooks/train_blip2_thorax_full.ipynb"
    },
    "Standard_NT": {
        "notebook": "notebooks/train_blip2_standard_nt.ipynb",
        "num_images": 1508,
        "output": "notebooks/train_blip2_standard_nt_full.ipynb"
    },
}

def train_category(category_name, config, num_epochs):
    """Train a single category"""
    print(f"\n{'='*70}")
    print(f"Training: {category_name}")
    print(f"Images: {config['num_images']}")
    print(f"Epochs: {num_epochs}")
    print(f"{'='*70}\n")

    # Run papermill
    cmd = [
        sys.executable, "-m", "papermill",
        config["notebook"],
        config["output"],
        "-p", "num_images", str(config["num_images"]),
        "-p", "num_epochs", str(num_epochs)
    ]

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per category
        )

        training_time = time.time() - start_time

        if result.returncode == 0:
            print(f"SUCCESS: {category_name} trained in {training_time/60:.1f} minutes\n")
            return {
                "category": category_name,
                "status": "success",
                "training_time_min": training_time / 60,
                "num_images": config["num_images"],
                "num_epochs": num_epochs
            }
        else:
            print(f"FAILED: {category_name}")
            print(f"Error: {result.stderr[-500:]}\n")  # Last 500 chars of error
            return {
                "category": category_name,
                "status": "failed",
                "error": result.stderr[-500:]
            }

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {category_name} exceeded 1 hour\n")
        return {
            "category": category_name,
            "status": "timeout"
        }
    except Exception as e:
        print(f"ERROR: {category_name} - {str(e)}\n")
        return {
            "category": category_name,
            "status": "error",
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="Train VQA models at full scale")
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--categories",
        type=str,
        default="all",
        help="Comma-separated list of categories or 'all' (default: all)"
    )
    parser.add_argument(
        "--sort-by-size",
        action="store_true",
        help="Train categories by size (smallest first)"
    )

    args = parser.parse_args()

    # Select categories
    if args.categories == "all":
        selected_categories = CATEGORIES
    else:
        selected_names = [c.strip() for c in args.categories.split(",")]
        selected_categories = {k: v for k, v in CATEGORIES.items() if k in selected_names}

    if not selected_categories:
        print("No valid categories selected!")
        sys.exit(1)

    # Sort by size if requested
    if args.sort_by_size:
        selected_categories = dict(sorted(
            selected_categories.items(),
            key=lambda x: x[1]["num_images"]
        ))

    print("="*70)
    print("Full-Scale VQA Training")
    print("="*70)
    print(f"Categories: {len(selected_categories)}")
    print(f"Epochs: {args.epochs}")
    print(f"Total images: {sum(c['num_images'] for c in selected_categories.values())}")
    print("="*70)

    # Confirm
    response = input("\nProceed with training? (y/N): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        sys.exit(0)

    # Train all categories
    results = []
    total_start = time.time()

    for category_name, config in selected_categories.items():
        result = train_category(category_name, config, args.epochs)
        results.append(result)

        # Clear CUDA cache between categories
        subprocess.run([sys.executable, "clear_cuda.py"], capture_output=True)
        time.sleep(5)  # Brief pause

    total_time = time.time() - total_start

    # Summary
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"\nResults:")

    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") != "success"]

    for result in successful:
        print(f"  ✓ {result['category']}: {result['training_time_min']:.1f}min")

    if failed:
        print(f"\nFailed:")
        for result in failed:
            print(f"  ✗ {result['category']}: {result.get('status', 'unknown error')}")

    print(f"\nSuccess rate: {len(successful)}/{len(results)}")

    # Save results
    output_file = Path("outputs/full_scale_training_results.json")
    output_file.parent.mkdir(exist_ok=True, parents=True)

    with open(output_file, 'w') as f:
        json.dump({
            "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_epochs": args.epochs,
            "total_time_hours": total_time / 3600,
            "categories_attempted": len(results),
            "categories_successful": len(successful),
            "results": results
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print("="*70)

if __name__ == "__main__":
    main()
