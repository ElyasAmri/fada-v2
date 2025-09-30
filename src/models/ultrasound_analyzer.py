"""
Ultrasound Analyzer Model
Modular architecture for fetal ultrasound analysis
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Optional


class UltrasoundAnalyzer(nn.Module):
    """
    Multi-task ultrasound analyzer with modular architecture

    Architecture:
    - Shared backbone (EfficientNet-B0)
    - Multiple task-specific heads:
        - Classification (organ type)
        - Abnormality detection
        - Attribute extraction (quality, orientation)
        - Future: Caption generation
    """

    def __init__(self,
                 num_classes: int = 5,
                 backbone_name: str = 'efficientnet_b0',
                 pretrained: bool = True):
        super(UltrasoundAnalyzer, self).__init__()

        self.num_classes = num_classes

        # Load backbone
        if backbone_name == 'efficientnet_b0':
            if pretrained:
                self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.efficientnet_b0(weights=None)
            # Get feature dimension
            feature_dim = self.backbone.classifier[1].in_features
            # Remove original classifier
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Task-specific heads
        self.heads = nn.ModuleDict({
            # Organ classification head
            'classification': nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            ),

            # Abnormality detection head
            'abnormality': nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1),
                nn.Sigmoid()
            ),

            # Quality assessment head
            'quality': nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 3)  # Good, Fair, Poor
            ),

            # Orientation detection head
            'orientation': nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 4)  # Axial, Sagittal, Coronal, Oblique
            )
        })

    def forward(self, x: torch.Tensor, tasks: Optional[list] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through backbone and specified task heads

        Args:
            x: Input tensor [batch_size, 3, 224, 224]
            tasks: List of tasks to compute (None = all tasks)

        Returns:
            Dictionary of task outputs
        """
        # Extract features from backbone
        features = self.backbone(x)

        # Determine which tasks to compute
        if tasks is None:
            tasks = list(self.heads.keys())

        # Compute outputs for each task
        outputs = {}
        for task in tasks:
            if task in self.heads:
                outputs[task] = self.heads[task](features)

        return outputs

    def freeze_backbone(self):
        """Freeze backbone parameters for transfer learning"""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """Get number of parameters in model"""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


class FeatureExtractor(nn.Module):
    """
    Standalone feature extractor (backbone only)
    Used for extracting features for downstream tasks
    """

    def __init__(self, backbone_name: str = 'efficientnet_b0', pretrained: bool = True):
        super(FeatureExtractor, self).__init__()

        if backbone_name == 'efficientnet_b0':
            if pretrained:
                self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.efficientnet_b0(weights=None)
            self.feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input"""
        return self.backbone(x)

    def freeze(self):
        """Freeze all parameters"""
        for param in self.parameters():
            param.requires_grad = False


if __name__ == "__main__":
    # Test the model
    model = UltrasoundAnalyzer(num_classes=5)

    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    # Forward pass
    outputs = model(dummy_input)

    print("Model Architecture Test:")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shapes:")
    for task, output in outputs.items():
        print(f"  {task}: {output.shape}")

    print(f"\nTotal parameters: {model.get_num_parameters():,}")
    print(f"Trainable parameters: {model.get_num_parameters(trainable_only=True):,}")