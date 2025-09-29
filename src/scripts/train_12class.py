"""
Train 12-class fetal ultrasound classifier with proper metrics and MLflow tracking
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import argparse
import json
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    cohen_kappa_score, classification_report, confusion_matrix
)

from src.data.dataset import FetalDataModule12Class
from src.data.augmentation import get_training_augmentation, get_validation_augmentation
from src.models.classifier import create_model
from src.utils.training import EarlyStopping, ModelCheckpoint
from src.utils.validation_metrics import ComprehensiveMetrics

import warnings
warnings.filterwarnings('ignore')


def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc='Training')
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Statistics
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_balanced_acc = balanced_accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc, epoch_balanced_acc, all_preds, all_labels


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)

    # Calculate comprehensive metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'balanced_accuracy': balanced_accuracy_score(all_labels, all_preds),
        'macro_f1': f1_score(all_labels, all_preds, average='macro'),
        'weighted_f1': f1_score(all_labels, all_preds, average='weighted'),
        'cohen_kappa': cohen_kappa_score(all_labels, all_preds)
    }

    return epoch_loss, metrics, all_preds, all_labels, all_probs


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Initialize data module
    print("\nInitializing data module...")
    data_module = FetalDataModule12Class(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_state=args.seed
    )

    # Setup with augmentations
    data_module.setup(
        train_transform=get_training_augmentation(args.image_size),
        val_transform=get_validation_augmentation(args.image_size)
    )

    # Get dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    # Create model and loss
    print(f"\nCreating model: {args.backbone}")
    model, criterion = create_model(
        num_classes=12,
        backbone=args.backbone,
        pretrained=args.pretrained,
        dropout_rate=args.dropout,
        class_weights=data_module.class_weights,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma
    )
    model = model.to(device)

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    # Mixed precision scaler
    scaler = GradScaler()

    # Initialize tracking
    early_stopping = EarlyStopping(patience=args.patience, mode='max', verbose=True)
    checkpoint = ModelCheckpoint(
        filepath=f'models/best_model_{args.backbone}_12class.pth',
        monitor='balanced_accuracy',
        mode='max',
        verbose=True
    )

    # MLflow tracking
    if args.use_mlflow:
        mlflow.set_experiment(args.experiment_name)
        mlflow.start_run()

        # Log parameters
        mlflow.log_params({
            'backbone': args.backbone,
            'pretrained': args.pretrained,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'epochs': args.epochs,
            'image_size': args.image_size,
            'use_focal_loss': args.use_focal_loss,
            'focal_gamma': args.focal_gamma,
            'dropout': args.dropout,
            'weight_decay': args.weight_decay,
            'num_classes': 12
        })

    # Training loop
    print("\nStarting training...")
    best_balanced_acc = 0.0

    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")

        # Train
        train_loss, train_acc, train_balanced_acc, train_preds, train_labels = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )

        # Validate
        val_loss, val_metrics, val_preds, val_labels, val_probs = validate(
            model, val_loader, criterion, device
        )

        # Update scheduler
        scheduler.step()

        # Print metrics
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Train Acc: {train_acc:.4f}, Train Balanced Acc: {train_balanced_acc:.4f}")
        print(f"\nVal Loss: {val_loss:.4f}")
        print(f"Val Metrics:")
        for metric_name, value in val_metrics.items():
            print(f"  {metric_name}: {value:.4f}")

        # MLflow logging
        if args.use_mlflow:
            mlflow.log_metrics({
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'train_balanced_accuracy': train_balanced_acc,
                'val_loss': val_loss,
                **{f'val_{k}': v for k, v in val_metrics.items()},
                'learning_rate': optimizer.param_groups[0]['lr']
            }, step=epoch)

        # Checkpoint and early stopping
        checkpoint(val_metrics['balanced_accuracy'], model, optimizer, epoch)

        if early_stopping(val_metrics['balanced_accuracy']):
            print("\nEarly stopping triggered!")
            break

        # Update best score
        if val_metrics['balanced_accuracy'] > best_balanced_acc:
            best_balanced_acc = val_metrics['balanced_accuracy']

    # Load best model and test
    print("\n" + "="*60)
    print("TESTING BEST MODEL")
    print("="*60)

    checkpoint_path = f'models/best_model_{args.backbone}_12class.pth'
    if Path(checkpoint_path).exists():
        checkpoint_dict = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint_dict['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint_dict['epoch']}")

    # Test evaluation
    test_loss, test_metrics, test_preds, test_labels, test_probs = validate(
        model, test_loader, criterion, device
    )

    print("\nTest Metrics:")
    for metric_name, value in test_metrics.items():
        print(f"  {metric_name}: {value:.4f}")

    # Detailed classification report
    print("\nClassification Report:")
    class_names = data_module.train_dataset.CLASSES
    print(classification_report(test_labels, test_preds, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    print("\nConfusion Matrix:")
    print(cm)

    # Log final metrics
    if args.use_mlflow:
        mlflow.log_metrics({
            f'test_{k}': v for k, v in test_metrics.items()
        })

        # Log model
        if args.log_model:
            mlflow.pytorch.log_model(model, "model")

        # Save classification report
        report_dict = classification_report(
            test_labels, test_preds,
            target_names=class_names,
            output_dict=True
        )
        with open('classification_report.json', 'w') as f:
            json.dump(report_dict, f, indent=2)
        mlflow.log_artifact('classification_report.json')

        mlflow.end_run()

    print("\nTraining complete!")
    print(f"Best validation balanced accuracy: {best_balanced_acc:.4f}")
    print(f"Test balanced accuracy: {test_metrics['balanced_accuracy']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train 12-class ultrasound classifier')

    # Data parameters
    parser.add_argument('--data-root', type=str, default='data/Fetal Ultrasound',
                        help='Path to data directory')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Input image size')

    # Model parameters
    parser.add_argument('--backbone', type=str, default='efficientnet_b0',
                        choices=['efficientnet_b0', 'resnet18', 'densenet121', 'mobilenet_v2'],
                        help='Backbone architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')

    # Loss parameters
    parser.add_argument('--use-focal-loss', action='store_true', default=True,
                        help='Use focal loss for imbalance')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='Focal loss gamma parameter')

    # Other parameters
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--use-mlflow', action='store_true', default=True,
                        help='Use MLflow tracking')
    parser.add_argument('--experiment-name', type=str, default='12-class-ultrasound',
                        help='MLflow experiment name')
    parser.add_argument('--log-model', action='store_true', default=False,
                        help='Log model to MLflow')

    args = parser.parse_args()
    main(args)