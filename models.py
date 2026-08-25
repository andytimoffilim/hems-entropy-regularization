import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class CIFARResNet18(nn.Module):
    """ResNet-18 adapted for CIFAR-10/100."""

    def __init__(self, embedding_dim: int = 128, num_classes: int = 10):
        super().__init__()

        # Load ResNet-18 without pretrained weights
        self.resnet = resnet18(weights=None)

        # Modify first conv layer for CIFAR (32x32 images)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()  # Remove maxpool

        # Get the final feature dimension
        # For ResNet-18, after all layers, the feature map is 512-d
        # But we need to determine the actual size

        # Remove the final classification layer
        self.resnet.fc = nn.Identity()

        # The feature dimension after global average pooling
        # For CIFAR-10 with our modifications, it's 512
        self.feature_dim = 512

        # Projection MLP
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, embedding_dim)
        )

        # Classifier
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        # Extract features
        features = self.resnet(x)

        # Ensure features are flattened
        if features.dim() > 2:
            features = features.view(features.size(0), -1)

        # Get embeddings
        embeddings = self.projection(features)

        # L2-normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Get logits
        logits = self.classifier(embeddings)

        return embeddings, logits


class MLP(nn.Module):
    """Simple MLP for tabular data."""

    def __init__(self, input_dim: int, embedding_dim: int = 128, num_classes: int = 10,
                 hidden_dims: list = [512, 256]):
        super().__init__()

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # Projection MLP
        self.projection = nn.Sequential(
            nn.Linear(prev_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, embedding_dim)
        )

        # Classifier
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        # Extract features
        features = self.backbone(x)

        # Get embeddings
        embeddings = self.projection(features)

        # L2-normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Get logits
        logits = self.classifier(embeddings)

        return embeddings, logits