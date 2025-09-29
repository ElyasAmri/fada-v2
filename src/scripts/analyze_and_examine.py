"""
Combined Dataset Analysis Tools
Merges functionality from analyze_dataset.py and examine_data.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from pathlib import Path
import pandas as pd
import numpy as np
import re
from collections import Counter
import argparse


def examine_data_structure(data_root: Path):
    """Examine the data folder structure and Excel files"""
    print("=" * 60)
    print("EXAMINING DATA STRUCTURE")
    print("=" * 60)

    # Examine folder structure
    for folder in data_root.iterdir():
        if not folder.is_dir():
            continue

        images = list(folder.glob('*.png')) + list(folder.glob('*.jpg'))
        if images:
            print(f"\n{folder.name}: {len(images)} images")
            # Sample filenames
            print(f"  Sample filenames (first 5):")
            for img in sorted(images)[:5]:
                print(f"    {img.name}")

            # Pattern analysis
            print(f"  Pattern analysis (first 20):")
            pattern_counts = Counter()
            for img in sorted(images)[:20]:
                if re.search(r'\d+', img.stem):
                    pattern_counts['has_number'] += 1
                if re.search(r'patient', img.stem, re.IGNORECASE):
                    pattern_counts['has_patient'] += 1
                if '_' in img.stem:
                    pattern_counts['has_underscore'] += 1

            for pattern, count in pattern_counts.items():
                print(f"    {pattern}: {count}/20")

    # Check Excel files
    print("\n" + "=" * 60)
    print("EXCEL FILES")
    print("=" * 60)

    excel_files = list(data_root.glob('*.xlsx')) + list(data_root.glob('*.xls'))
    if excel_files:
        for excel_file in excel_files[:3]:  # Limit to first 3
            print(f"\nFound: {excel_file.name}")
            try:
                df = pd.read_excel(excel_file)
                print(f"  Shape: {df.shape}")
                print(f"  Columns: {list(df.columns)}")

                # Check if there's actual data
                non_null_counts = df.notna().sum().sum()
                if non_null_counts > len(df):  # More than just the image names column
                    print(f"  Contains data: {non_null_counts} non-null values")
                else:
                    print(f"  Empty template: Only image names filled")

                # Show first few rows
                print(f"\n  First 3 rows:")
                print(df.head(3))
            except Exception as e:
                print(f"  Error reading: {e}")
    else:
        print("No Excel files found")


def analyze_class_imbalance(data_root: Path):
    """Analyze class distribution and imbalance"""
    from src.data.dataset import FetalUltrasoundDataset, FetalDataModule
    from src.data.augmentation import get_validation_augmentation
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt

    print("\n" + "=" * 60)
    print("CLASS IMBALANCE ANALYSIS")
    print("=" * 60)

    # Load data module
    data_module = FetalDataModule(
        data_root=str(data_root),
        batch_size=32,
        num_workers=0,
        patient_aware_split=True
    )
    data_module.setup()

    # Get datasets
    train_dataset = data_module.train_dataset
    val_dataset = data_module.val_dataset
    test_dataset = data_module.test_dataset

    # Analyze distributions
    print("\nCLASS DISTRIBUTION:")
    print("-" * 40)

    splits = {
        'Train': train_dataset.labels,
        'Val': val_dataset.labels,
        'Test': test_dataset.labels,
        'Full': train_dataset.labels + val_dataset.labels + test_dataset.labels
    }

    classes = train_dataset.CLASSES

    for split_name, labels in splits.items():
        print(f"\n{split_name} Set ({len(labels)} samples):")
        label_counts = Counter(labels)

        for idx, class_name in enumerate(classes):
            count = label_counts.get(idx, 0)
            percentage = (count / len(labels)) * 100 if labels else 0
            print(f"  {class_name}: {count:4d} ({percentage:5.1f}%)")

        # Imbalance ratio
        if label_counts:
            max_count = max(label_counts.values())
            min_count = min(label_counts.values()) if min(label_counts.values()) > 0 else 1
            imbalance_ratio = max_count / min_count
            print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")

    # Check for missing classes
    print("\n" + "-" * 40)
    print("CRITICAL ISSUES:")

    issues_found = False
    for split_name in ['Train', 'Val', 'Test']:
        labels = splits[split_name]
        label_set = set(labels)
        missing_classes = []
        for idx, class_name in enumerate(classes):
            if idx not in label_set:
                missing_classes.append(class_name)
                issues_found = True

        if missing_classes:
            print(f"WARNING: {split_name} set missing classes: {missing_classes}")

    if not issues_found:
        print("No missing classes detected")

    # Patient ID analysis
    print("\n" + "-" * 40)
    print("PATIENT ID ANALYSIS:")

    print(f"Train patients: {len(set(train_dataset.patient_ids))}")
    print(f"Val patients: {len(set(val_dataset.patient_ids))}")
    print(f"Test patients: {len(set(test_dataset.patient_ids))}")

    # Check for patient overlap
    train_patients = set(train_dataset.patient_ids)
    val_patients = set(val_dataset.patient_ids)
    test_patients = set(test_dataset.patient_ids)

    train_val_overlap = train_patients & val_patients
    train_test_overlap = train_patients & test_patients
    val_test_overlap = val_patients & test_patients

    if train_val_overlap or train_test_overlap or val_test_overlap:
        print("\nWARNING: Patient overlap detected!")
        if train_val_overlap:
            print(f"  Train-Val overlap: {len(train_val_overlap)} patients")
        if train_test_overlap:
            print(f"  Train-Test overlap: {len(train_test_overlap)} patients")
        if val_test_overlap:
            print(f"  Val-Test overlap: {len(val_test_overlap)} patients")
    else:
        print("Good: No patient overlap between splits")

    # Sample patient IDs to understand their structure
    print("\nSample patient IDs (first 5 from each split):")
    for split_name, dataset in [('Train', train_dataset), ('Val', val_dataset), ('Test', test_dataset)]:
        print(f"  {split_name}: {list(sorted(set(dataset.patient_ids)))[:5]}")

    return splits, classes


def main():
    parser = argparse.ArgumentParser(description='Analyze dataset structure and class distribution')
    parser.add_argument('--data-root', type=str, default='data/Fetal Ultrasound',
                        help='Path to data root directory')
    parser.add_argument('--mode', choices=['examine', 'imbalance', 'both'], default='both',
                        help='Analysis mode')
    args = parser.parse_args()

    data_root = Path(args.data_root)

    if not data_root.exists():
        print(f"ERROR: Data root not found: {data_root}")
        return

    if args.mode in ['examine', 'both']:
        examine_data_structure(data_root)

    if args.mode in ['imbalance', 'both']:
        analyze_class_imbalance(data_root)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()