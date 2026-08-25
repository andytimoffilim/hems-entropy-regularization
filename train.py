import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from typing import Dict, Any, List
import os
import glob


class Trainer:
    """Trainer for CIFAR experiments."""

    def __init__(self, model: nn.Module, loss_fn: nn.Module, config: Dict[str, Any],
                 save_dir: str = './checkpoints'):
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self.save_dir = save_dir

        # Optimizer
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=config.get('learning_rate', 0.1),
            momentum=config.get('momentum', 0.9),
            weight_decay=config.get('weight_decay', 5e-4)
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 200),
            eta_min=1e-5
        )

        self.best_val_f1 = 0.0
        self.best_checkpoint_path = None
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }

        os.makedirs(save_dir, exist_ok=True)

    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        loss_dict = {}

        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(self.config['device'])
            labels = labels.to(self.config['device'])

            # Forward pass
            embeddings, logits = self.model(data)

            # Compute loss
            if hasattr(self.loss_fn, 'step'):
                self.loss_fn.step()

            loss, loss_dict = self.loss_fn(embeddings, labels, logits)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        return avg_loss, loss_dict

    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0

        with torch.no_grad():
            for data, labels in val_loader:
                data = data.to(self.config['device'])
                labels = labels.to(self.config['device'])

                embeddings, logits = self.model(data)

                # Compute loss
                if hasattr(self.loss_fn, 'step'):
                    self.loss_fn.step()

                loss, _ = self.loss_fn(embeddings, labels, logits)
                total_loss += loss.item()

                # Get predictions
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)

        # Compute metrics
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')

        return avg_loss, acc, f1

    def train(self, train_loader, val_loader, epochs: int):
        """Full training loop."""
        for epoch in range(1, epochs + 1):
            # Train
            train_loss, loss_dict = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc, val_f1 = self.validate(val_loader)

            # Step scheduler
            self.scheduler.step()

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)

            # Print progress every 10 epochs
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch}/{epochs}: "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_acc:.4f}, "
                      f"Val F1: {val_f1:.4f}")

            # Save best model
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_checkpoint_path = self.save_checkpoint(epoch, val_f1)

    def save_checkpoint(self, epoch: int, val_f1: float) -> str:
        """Save model checkpoint and return path."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_f1': val_f1,
            'history': self.history,
            'config': self.config
        }

        path = os.path.join(self.save_dir, f'best_model_epoch_{epoch}_f1_{val_f1:.4f}.pth')
        torch.save(checkpoint, path)
        return path

    def get_best_checkpoint(self) -> str:
        """Get the best checkpoint path."""
        return self.best_checkpoint_path

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint {checkpoint_path} not found.")
            return None

        checkpoint = torch.load(checkpoint_path, map_location=self.config['device'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_f1 = checkpoint['val_f1']
        self.history = checkpoint['history']
        return checkpoint['epoch']

    def get_best_checkpoint_path(self) -> str:
        """Get best checkpoint path."""
        if self.best_checkpoint_path:
            return self.best_checkpoint_path

        # Try to find the best checkpoint in directory
        checkpoint_files = glob.glob(os.path.join(self.save_dir, 'best_model_*.pth'))
        if checkpoint_files:
            # Get the one with highest f1
            def extract_f1(path):
                try:
                    return float(path.split('f1_')[-1].replace('.pth', ''))
                except:
                    return 0.0

            best_file = max(checkpoint_files, key=extract_f1)
            return best_file

        return None