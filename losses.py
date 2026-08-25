import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class UniformityLoss(nn.Module):
    """Encourage uniform distribution of embeddings on hypersphere."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings):
        """
        Args:
            embeddings: L2-normalized embeddings [batch_size, embedding_dim]
        Returns:
            uniformity_loss: scalar
        """
        batch_size = embeddings.size(0)

        # Compute pairwise cosine similarity
        sim_matrix = torch.mm(embeddings, embeddings.T)  # [batch_size, batch_size]

        # Exclude self-similarity
        mask = torch.eye(batch_size, device=embeddings.device)
        sim_matrix = sim_matrix * (1 - mask)  # Zero out diagonal

        # Compute loss
        loss = torch.logsumexp(sim_matrix / self.temperature, dim=1).mean()

        return loss


class DiversityLoss(nn.Module):
    """Prevent dimensional collapse by controlling covariance."""

    def __init__(self, gamma: float = 0.1):
        super().__init__()
        self.gamma = gamma

    def forward(self, embeddings):
        """
        Args:
            embeddings: L2-normalized embeddings [batch_size, embedding_dim]
        Returns:
            diversity_loss: scalar
        """
        batch_size, emb_dim = embeddings.size()

        # Center the embeddings
        centered = embeddings - embeddings.mean(dim=0, keepdim=True)

        # Compute covariance matrix
        sigma = torch.mm(centered.T, centered) / (batch_size - 1)

        # Get diagonal
        diag = torch.diag(sigma)

        # Compute loss
        std_diag = diag.std()
        mean_diag = diag.mean()

        # Avoid division by zero
        if mean_diag < 1e-8:
            mean_diag = 1e-8

        # Variance loss (encourages equal variance across dimensions)
        variance_loss = std_diag / mean_diag

        # Correlation loss (encourages identity covariance)
        # Create identity matrix
        eye = torch.eye(emb_dim, device=embeddings.device)
        correlation_loss = torch.norm(sigma - eye, p='fro') ** 2

        return variance_loss + self.gamma * correlation_loss


class SeparationLoss(nn.Module):
    """Encourage inter-class separation via class centroids."""

    def __init__(self, num_classes: int, momentum: float = 0.99):
        super().__init__()
        self.num_classes = num_classes
        self.momentum = momentum
        self.centroids = None
        self.centroid_counts = None

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: L2-normalized embeddings [batch_size, embedding_dim]
            labels: class labels [batch_size]
        Returns:
            separation_loss: scalar
        """
        batch_size, emb_dim = embeddings.size()
        device = embeddings.device

        # Initialize centroids if not exists
        if self.centroids is None:
            self.centroids = torch.zeros(self.num_classes, emb_dim, device=device)
            self.centroid_counts = torch.zeros(self.num_classes, device=device)

        # Update centroids (without gradient - use detach)
        with torch.no_grad():
            unique_labels = torch.unique(labels)
            for label in unique_labels:
                mask = (labels == label)
                class_embeddings = embeddings[mask]
                if len(class_embeddings) > 0:
                    current_centroid = class_embeddings.mean(dim=0)

                    if self.centroid_counts[label] == 0:
                        self.centroids[label] = current_centroid
                    else:
                        self.centroids[label] = self.momentum * self.centroids[label] + \
                                                (1 - self.momentum) * current_centroid

                    self.centroid_counts[label] += 1

        # Compute separation loss (with gradient)
        seen_classes = torch.unique(labels)

        if len(seen_classes) < 2:
            return torch.tensor(0.0, device=device)

        # Get centroids for seen classes (detach to avoid gradient issues)
        seen_centroids = self.centroids[seen_classes].detach()

        # Compute pairwise distances
        dist_matrix = torch.cdist(seen_centroids, seen_centroids, p=2)

        # Exclude self-distances
        mask = torch.eye(len(seen_classes), device=device) == 0
        min_distance = dist_matrix[mask].min()

        # Hinge loss: penalize if distance < 1
        loss = F.relu(1 - min_distance)

        return loss


class HEMSLoss(nn.Module):
    """Complete HEMS loss function."""

    def __init__(self, num_classes: int, temperature: float = 0.1,
                 lambda_u_max: float = 0.05, lambda_d: float = 0.01,
                 lambda_s: float = 0.02, warmup_epochs: int = 20,
                 ema_momentum: float = 0.99):
        super().__init__()

        # Component losses
        self.uniformity_loss = UniformityLoss(temperature)
        self.diversity_loss = DiversityLoss()
        self.separation_loss = SeparationLoss(num_classes, ema_momentum)

        # Weights
        self.lambda_u_max = lambda_u_max
        self.lambda_d = lambda_d
        self.lambda_s = lambda_s
        self.warmup_epochs = warmup_epochs

        self.current_epoch = 0
        self.num_classes = num_classes

    def step(self):
        """Advance to next epoch."""
        self.current_epoch += 1

    def get_lambda_u(self):
        """Get current lambda_u based on warmup schedule."""
        if self.current_epoch < self.warmup_epochs:
            return self.lambda_u_max * (self.current_epoch / self.warmup_epochs)
        else:
            return self.lambda_u_max

    def forward(self, embeddings, labels, logits):
        """
        Args:
            embeddings: L2-normalized embeddings [batch_size, embedding_dim]
            labels: class labels [batch_size]
            logits: classifier logits [batch_size, num_classes]
        Returns:
            total_loss: scalar
            loss_dict: dictionary with individual losses
        """
        # Cross-entropy loss
        ce_loss = F.cross_entropy(logits, labels)

        # Uniformity loss
        u_loss = self.uniformity_loss(embeddings)
        lambda_u = self.get_lambda_u()

        # Diversity loss
        d_loss = self.diversity_loss(embeddings)

        # Separation loss
        s_loss = self.separation_loss(embeddings, labels)

        # Total loss
        total_loss = ce_loss + lambda_u * u_loss + \
                     self.lambda_d * d_loss + self.lambda_s * s_loss

        return total_loss, {
            'ce_loss': ce_loss.item(),
            'uniformity_loss': u_loss.item(),
            'diversity_loss': d_loss.item(),
            'separation_loss': s_loss.item(),
            'lambda_u': lambda_u
        }


class BaselineLoss(nn.Module):
    """Baseline loss (only cross-entropy)."""

    def __init__(self):
        super().__init__()

    def forward(self, embeddings, labels, logits):
        ce_loss = F.cross_entropy(logits, labels)
        return ce_loss, {'ce_loss': ce_loss.item()}


def create_loss(name: str, num_classes: int, **kwargs):
    """Factory function to create loss."""
    if name == 'baseline':
        return BaselineLoss()
    elif name == 'hems':
        return HEMSLoss(num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown loss: {name}")