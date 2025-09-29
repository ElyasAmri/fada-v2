"""
Comprehensive Validation Metrics for Medical Image Classification
Includes confusion matrix, per-class metrics, and statistical significance testing
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    cohen_kappa_score
)
import torch
from typing import List, Tuple, Dict, Optional
import pandas as pd
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComprehensiveMetrics:
    """Calculate and visualize comprehensive metrics for model evaluation"""

    def __init__(self, class_names: List[str]):
        """
        Initialize metrics calculator

        Args:
            class_names: List of class names for labeling
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.reset()

    def reset(self):
        """Reset all stored predictions"""
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []

    def update(self, preds: torch.Tensor, labels: torch.Tensor, probs: Optional[torch.Tensor] = None):
        """
        Update with batch predictions

        Args:
            preds: Predicted class indices
            labels: True class indices
            probs: Optional prediction probabilities for AUC
        """
        self.all_preds.extend(preds.detach().cpu().numpy())
        self.all_labels.extend(labels.detach().cpu().numpy())
        if probs is not None:
            self.all_probs.extend(probs.detach().cpu().numpy())

    def compute_metrics(self) -> Dict:
        """
        Compute all metrics

        Returns:
            Dictionary containing all computed metrics
        """
        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_preds)

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(self.num_classes)
        )

        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )

        # Macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )

        # Cohen's Kappa (agreement metric)
        kappa = cohen_kappa_score(y_true, y_pred)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))

        # AUC if probabilities available
        auc_scores = None
        if self.all_probs:
            try:
                y_probs = np.array(self.all_probs)
                if self.num_classes == 2:
                    auc_scores = roc_auc_score(y_true, y_probs[:, 1])
                else:
                    # One-vs-rest AUC for multiclass
                    from sklearn.preprocessing import label_binarize
                    y_true_binarized = label_binarize(y_true, classes=range(self.num_classes))
                    auc_scores = {}
                    for i in range(self.num_classes):
                        try:
                            auc_scores[self.class_names[i]] = roc_auc_score(
                                y_true_binarized[:, i], y_probs[:, i]
                            )
                        except:
                            auc_scores[self.class_names[i]] = None
            except Exception as e:
                logger.warning(f"Could not compute AUC: {e}")

        metrics = {
            'accuracy': accuracy,
            'kappa': kappa,
            'confusion_matrix': cm,
            'per_class': {
                'precision': dict(zip(self.class_names, precision)),
                'recall': dict(zip(self.class_names, recall)),
                'f1': dict(zip(self.class_names, f1)),
                'support': dict(zip(self.class_names, support))
            },
            'weighted': {
                'precision': precision_weighted,
                'recall': recall_weighted,
                'f1': f1_weighted
            },
            'macro': {
                'precision': precision_macro,
                'recall': recall_macro,
                'f1': f1_macro
            },
            'auc': auc_scores
        }

        return metrics

    def plot_confusion_matrix(self, save_path: Optional[str] = None, normalize: bool = False):
        """
        Plot confusion matrix with detailed annotations

        Args:
            save_path: Path to save figure
            normalize: Whether to normalize values to percentages
        """
        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_preds)

        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))

        if normalize:
            # Handle division by zero for classes with no samples
            cm = cm.astype('float')
            cm_sum = cm.sum(axis=1)[:, np.newaxis]
            # Replace zeros with 1 to avoid division by zero
            cm_sum[cm_sum == 0] = 1
            cm = cm / cm_sum * 100

        plt.figure(figsize=(10, 8))

        # Create heatmap
        # Handle NaN values in confusion matrix
        if normalize:
            # Replace NaN with 0 for visualization
            cm = np.nan_to_num(cm, nan=0.0)
            fmt = '.1f'  # Remove % from format, will add manually
        else:
            fmt = 'd'

        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   square=True,
                   cbar_kws={'label': 'Percentage' if normalize else 'Count'})

        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        # Add accuracy for each class
        for i in range(self.num_classes):
            class_acc = cm[i, i] / cm[i].sum() * 100 if not normalize else cm[i, i]
            plt.text(self.num_classes + 0.3, i + 0.5, f'{class_acc:.1f}%',
                    ha='left', va='center')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")

        plt.show()

    def plot_per_class_metrics(self, save_path: Optional[str] = None):
        """
        Plot per-class precision, recall, and F1 scores

        Args:
            save_path: Path to save figure
        """
        metrics = self.compute_metrics()
        per_class = metrics['per_class']

        # Prepare data for plotting
        classes = self.class_names
        precision_values = [per_class['precision'][c] for c in classes]
        recall_values = [per_class['recall'][c] for c in classes]
        f1_values = [per_class['f1'][c] for c in classes]

        x = np.arange(len(classes))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))

        # Create bars
        bars1 = ax.bar(x - width, precision_values, width, label='Precision')
        bars2 = ax.bar(x, recall_values, width, label='Recall')
        bars3 = ax.bar(x + width, f1_values, width, label='F1-Score')

        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')

        ax.set_xlabel('Classes')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.set_ylim([0, 1.1])

        # Add horizontal line for average
        ax.axhline(y=metrics['weighted']['f1'], color='r', linestyle='--',
                  label=f"Weighted F1: {metrics['weighted']['f1']:.3f}")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Per-class metrics saved to {save_path}")

        plt.show()

    def print_classification_report(self):
        """Print detailed classification report"""
        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_preds)

        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(y_true, y_pred, target_names=self.class_names, digits=3))

        metrics = self.compute_metrics()
        print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
        print(f"Cohen's Kappa: {metrics['kappa']:.4f}")

        if metrics['auc'] is not None:
            if isinstance(metrics['auc'], dict):
                print("\nAUC Scores:")
                for class_name, auc in metrics['auc'].items():
                    if auc is not None:
                        print(f"  {class_name}: {auc:.4f}")
            else:
                print(f"AUC Score: {metrics['auc']:.4f}")

    def statistical_significance_test(self,
                                     other_preds: List[int],
                                     test_type: str = 'mcnemar') -> Dict:
        """
        Test statistical significance between two models

        Args:
            other_preds: Predictions from another model
            test_type: Type of test ('mcnemar' or 'wilcoxon')

        Returns:
            Dictionary with test results
        """
        y_true = np.array(self.all_labels)
        y_pred1 = np.array(self.all_preds)
        y_pred2 = np.array(other_preds)

        if test_type == 'mcnemar':
            # McNemar's test for paired nominal data
            # Create contingency table
            correct1_correct2 = np.sum((y_pred1 == y_true) & (y_pred2 == y_true))
            correct1_wrong2 = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))
            wrong1_correct2 = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))
            wrong1_wrong2 = np.sum((y_pred1 != y_true) & (y_pred2 != y_true))

            # Perform McNemar's test
            from statsmodels.stats.contingency_tables import mcnemar
            table = [[correct1_correct2, correct1_wrong2],
                    [wrong1_correct2, wrong1_wrong2]]
            result = mcnemar(table, exact=False, correction=True)

            return {
                'test': 'McNemar',
                'statistic': result.statistic,
                'p_value': result.pvalue,
                'contingency_table': table,
                'significant': result.pvalue < 0.05
            }

        elif test_type == 'wilcoxon':
            # Wilcoxon signed-rank test
            acc1 = (y_pred1 == y_true).astype(int)
            acc2 = (y_pred2 == y_true).astype(int)
            statistic, p_value = stats.wilcoxon(acc1, acc2)

            return {
                'test': 'Wilcoxon signed-rank',
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < 0.05
            }

        else:
            raise ValueError(f"Unknown test type: {test_type}")

    def export_results(self, save_path: str):
        """
        Export all metrics to a CSV file

        Args:
            save_path: Path to save CSV
        """
        metrics = self.compute_metrics()

        # Create dataframe with results
        results = []

        for class_name in self.class_names:
            results.append({
                'Class': class_name,
                'Precision': metrics['per_class']['precision'][class_name],
                'Recall': metrics['per_class']['recall'][class_name],
                'F1-Score': metrics['per_class']['f1'][class_name],
                'Support': metrics['per_class']['support'][class_name],
                'AUC': metrics['auc'][class_name] if isinstance(metrics['auc'], dict) else None
            })

        # Add overall metrics
        results.append({
            'Class': 'Weighted Average',
            'Precision': metrics['weighted']['precision'],
            'Recall': metrics['weighted']['recall'],
            'F1-Score': metrics['weighted']['f1'],
            'Support': len(self.all_labels),
            'AUC': None
        })

        results.append({
            'Class': 'Macro Average',
            'Precision': metrics['macro']['precision'],
            'Recall': metrics['macro']['recall'],
            'F1-Score': metrics['macro']['f1'],
            'Support': len(self.all_labels),
            'AUC': None
        })

        df = pd.DataFrame(results)
        df.to_csv(save_path, index=False)
        logger.info(f"Results exported to {save_path}")

        return df