import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split
import os
from typing import Tuple, List, Dict


class CIFAR10DataLoader:
    """CIFAR-10 data loader with controlled stress regimes."""

    def __init__(self, data_root: str, batch_size: int = 128, num_workers: int = 4):
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Standard CIFAR-10 normalization
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2023, 0.1994, 0.2010)

        # Base transforms for training (without augmentation)
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        # Augmented transforms for training
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        # Test transform (no augmentation)
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def load_full_data(self) -> Tuple[Dataset, Dataset]:
        """Load full CIFAR-10 train and test sets."""
        train_set = torchvision.datasets.CIFAR10(
            root=self.data_root, train=True, download=True,
            transform=self.train_transform
        )

        test_set = torchvision.datasets.CIFAR10(
            root=self.data_root, train=False, download=True,
            transform=self.test_transform
        )

        return train_set, test_set

    def create_validation_split(self, train_set: Dataset, val_size: int = 5000) -> Tuple[Subset, Subset]:
        """Create stratified validation split from training set."""
        # Get labels
        labels = [train_set.targets[i] for i in range(len(train_set))]

        # Split indices
        train_indices, val_indices = train_test_split(
            range(len(train_set)),
            test_size=val_size,
            stratify=labels,
            random_state=42
        )

        train_subset = Subset(train_set, train_indices)
        val_subset = Subset(train_set, val_indices)

        return train_subset, val_subset

    def create_label_scarcity_split(self, train_subset: Subset, fraction: float, seed: int) -> Subset:
        """Create stratified subset with reduced training labels."""
        # Get indices and labels from the subset
        indices = train_subset.indices
        labels = [train_subset.dataset.targets[i] for i in indices]

        # Calculate number of samples per class
        num_samples = int(len(indices) * fraction)

        # Stratified sampling
        selected_indices = []
        for class_id in range(10):
            class_indices = [i for i, label in enumerate(labels) if label == class_id]
            num_class_samples = int(len(class_indices) * fraction)
            selected_class_indices = np.random.RandomState(seed).choice(
                class_indices,
                num_class_samples,
                replace=False
            )
            selected_indices.extend(selected_class_indices)

        # Create new subset
        original_indices = [indices[i] for i in selected_indices]
        return Subset(train_subset.dataset, original_indices)

    def create_long_tailed_split(self, train_subset: Subset, imbalance_ratio: int, seed: int) -> Subset:
        """Create long-tailed distribution for class imbalance experiments."""
        indices = train_subset.indices
        labels = [train_subset.dataset.targets[i] for i in indices]

        # Count samples per class
        class_counts = {}
        for label in labels:
            class_counts[label] = class_counts.get(label, 0) + 1

        # Shuffle class order for randomness
        classes = list(range(10))
        rng = np.random.RandomState(seed)
        rng.shuffle(classes)

        # Assign samples using exponential long-tail schedule
        # n_c = n_max * IR^(-c/(C-1))
        C = 10
        n_max = max(class_counts.values())

        selected_indices = []
        for i, (idx, label) in enumerate(zip(indices, labels)):
            # Find position of this class in shuffled order
            class_pos = classes.index(label)
            target_count = int(n_max * (imbalance_ratio ** (-class_pos / (C - 1))))

            # Keep only first target_count samples for this class
            class_selected = sum(1 for j, l in enumerate(labels[:i + 1]) if l == label)
            if class_selected <= target_count:
                selected_indices.append(idx)

        return Subset(train_subset.dataset, selected_indices)

    def get_dataloaders(self, train_dataset: Dataset, val_dataset: Dataset,
                        test_dataset: Dataset) -> Dict[str, DataLoader]:
        """Create data loaders for train, validation, and test sets."""
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )

        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }


class CIFAR100DataLoader(CIFAR10DataLoader):
    """CIFAR-100 data loader (inherits from CIFAR-10)."""

    def load_full_data(self) -> Tuple[Dataset, Dataset]:
        """Load full CIFAR-100 train and test sets."""
        train_set = torchvision.datasets.CIFAR100(
            root=self.data_root, train=True, download=True,
            transform=self.train_transform
        )

        test_set = torchvision.datasets.CIFAR100(
            root=self.data_root, train=False, download=True,
            transform=self.test_transform
        )

        return train_set, test_set

    def create_long_tailed_split(self, train_subset: Subset, imbalance_ratio: int, seed: int) -> Subset:
        """Create long-tailed distribution for CIFAR-100 (100 classes)."""
        indices = train_subset.indices
        labels = [train_subset.dataset.targets[i] for i in indices]

        # Count samples per class
        class_counts = {}
        for label in labels:
            class_counts[label] = class_counts.get(label, 0) + 1

        # Shuffle class order for randomness
        classes = list(range(100))
        rng = np.random.RandomState(seed)
        rng.shuffle(classes)

        # Use exponential long-tail schedule
        C = 100
        n_max = max(class_counts.values())

        selected_indices = []
        for i, (idx, label) in enumerate(zip(indices, labels)):
            class_pos = classes.index(label)
            target_count = int(n_max * (imbalance_ratio ** (-class_pos / (C - 1))))

            class_selected = sum(1 for j, l in enumerate(labels[:i + 1]) if l == label)
            if class_selected <= target_count:
                selected_indices.append(idx)

        return Subset(train_subset.dataset, selected_indices)