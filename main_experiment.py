import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import pickle
from datetime import datetime
import json
import time

from config import Config
from data_loader import CIFAR10DataLoader, CIFAR100DataLoader
from models import CIFARResNet18
from losses import create_loss
from train import Trainer
from evaluate import EmbeddingAnalyzer


class OptimizedExperimentRunner:
    """Optimized runner with dynamic epochs and intelligent experiment selection."""

    def __init__(self, config: Config):
        self.config = config
        self.results = []
        self.device = config.device
        self.start_time = time.time()

        # Create directories
        os.makedirs('./results', exist_ok=True)
        os.makedirs('./checkpoints', exist_ok=True)
        os.makedirs('./embeddings', exist_ok=True)

        # Save config
        with open('config_saved.json', 'w') as f:
            # Convert dataclass to dict
            config_dict = {k: v for k, v in config.__dict__.items()
                           if not k.startswith('_')}
            json.dump(config_dict, f, indent=2)

        # Create data loaders
        self.cifar10_loader = CIFAR10DataLoader(
            config.data_root, config.batch_size
        )
        self.cifar100_loader = CIFAR100DataLoader(
            config.data_root, config.batch_size
        )

        # Create analyzer
        self.analyzer = EmbeddingAnalyzer(
            config.embedding_dim, config.num_centroids
        )

        # Track experiments
        self.completed = 0
        self.total = 0

    def save_checkpoint(self, checkpoint_name):
        """Save full state to checkpoint."""
        checkpoint = {
            'results': self.results,
            'completed': self.completed,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'elapsed_time': time.time() - self.start_time
        }
        with open(f'./checkpoints/checkpoint_{checkpoint_name}.pkl', 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"💾 Checkpoint saved: {checkpoint_name} ({self.completed} experiments)")

    def load_checkpoint(self, checkpoint_name):
        """Load state from checkpoint."""
        with open(f'./checkpoints/checkpoint_{checkpoint_name}.pkl', 'rb') as f:
            checkpoint = pickle.load(f)
        self.results = checkpoint['results']
        self.completed = checkpoint['completed']
        print(f"📂 Checkpoint loaded: {checkpoint_name} ({len(self.results)} results)")

    def save_intermediate_results(self):
        """Save results after each experiment."""
        if self.results:
            df = pd.DataFrame(self.results)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.results_file = f'./results/results_intermediate_{timestamp}.csv'
            df.to_csv(self.results_file, index=False)

    def save_final_results(self):
        """Save final results."""
        if self.results:
            df = pd.DataFrame(self.results)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'./results/results_final_{timestamp}.csv'
            df.to_csv(filename, index=False)
            print(f"\n✅ Final results saved to {filename}")
            return df
        return None

    def run_experiment(self, dataset_name, regime_name, fraction, imbalance_ratio,
                       seeds, num_classes, dataloader, epochs):
        """Run experiments for a specific dataset and regime."""
        print(f"\n{'=' * 60}")
        print(f"Running: {dataset_name} - {regime_name}")
        print(f"Epochs: {epochs}")
        print(f"Seeds: {seeds}")
        print(f"{'=' * 60}")

        # Load data
        print(f"Loading {dataset_name} data...")
        train_set, test_set = dataloader.load_full_data()
        train_subset, val_subset = dataloader.create_validation_split(train_set)

        for seed in seeds:
            print(f"\n--- Seed: {seed} ---")

            # Create dataset based on regime
            if 'scarcity' in regime_name:
                train_dataset = dataloader.create_label_scarcity_split(
                    train_subset, fraction, seed
                )
            elif 'longtail' in regime_name:
                train_dataset = dataloader.create_long_tailed_split(
                    train_subset, imbalance_ratio, seed
                )
            else:  # balanced
                train_dataset = train_subset

            # Create dataloaders
            dataloaders = dataloader.get_dataloaders(
                train_dataset, val_subset, test_set
            )

            # Run Baseline and HEMS
            for method in ['baseline', 'hems']:
                self.completed += 1
                print(f"\n  [{self.completed}/{self.total}] Method: {method}")

                # Create model
                model = CIFARResNet18(
                    embedding_dim=self.config.embedding_dim,
                    num_classes=num_classes
                ).to(self.device)

                # Create loss
                if method == 'baseline':
                    loss_fn = create_loss('baseline', num_classes)
                else:
                    loss_fn = create_loss('hems', num_classes,
                                          temperature=self.config.temperature,
                                          lambda_u_max=self.config.lambda_u_max,
                                          lambda_d=self.config.lambda_d,
                                          lambda_s=self.config.lambda_s,
                                          warmup_epochs=min(self.config.warmup_epochs, epochs // 5),
                                          ema_momentum=self.config.ema_momentum)

                # Create trainer
                save_dir = f"./checkpoints/{dataset_name}/{regime_name}/{method}/seed_{seed}"
                trainer_config = {
                    'device': self.device,
                    'learning_rate': self.config.learning_rate,
                    'momentum': self.config.momentum,
                    'weight_decay': self.config.weight_decay,
                    'epochs': epochs
                }
                trainer = Trainer(model, loss_fn, trainer_config, save_dir)

                # Train
                print(f"  Training {method} on {regime_name} (seed={seed})...")
                start_time = time.time()
                trainer.train(
                    dataloaders['train'],
                    dataloaders['val'],
                    epochs
                )
                train_time = time.time() - start_time
                print(f"  Training time: {train_time / 60:.1f} minutes")

                # Load best model
                best_checkpoint = trainer.get_best_checkpoint_path()
                if best_checkpoint:
                    trainer.load_checkpoint(best_checkpoint)

                # Extract embeddings and compute metrics
                print("  Computing geometry metrics...")
                train_embeddings, train_labels = self.analyzer.extract_embeddings(
                    model, dataloaders['train'], self.device
                )
                test_embeddings, test_labels = self.analyzer.extract_embeddings(
                    model, dataloaders['test'], self.device
                )

                # Compute geometry metrics
                geometry_metrics = self.analyzer.compute_all_metrics(
                    train_embeddings, test_embeddings,
                    train_labels, test_labels
                )

                # Get final performance
                _, val_acc, val_f1 = trainer.validate(dataloaders['val'])

                # Save results
                result = {
                    'dataset': dataset_name,
                    'regime': regime_name,
                    'method': method,
                    'seed': seed,
                    'epochs': epochs,
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    **geometry_metrics
                }
                self.results.append(result)

                # Save intermediate results after each experiment
                self.save_intermediate_results()

                # Save checkpoint every 5 experiments
                if self.completed % 5 == 0:
                    self.save_checkpoint(f"progress_{self.completed}")

                # Save embeddings (only for critical experiments)
                if self._is_critical_experiment(dataset_name, regime_name, method):
                    self.save_embeddings(
                        train_embeddings, test_embeddings,
                        train_labels, test_labels,
                        dataset_name, regime_name, method, seed
                    )

                print(f"  ✅ Completed {method} on {regime_name} (seed={seed})")
                print(f"     Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
                print(f"     Effective Rank: {geometry_metrics['effective_rank']:.2f}")
                print(f"     R_hat: {geometry_metrics['R_hat']:.4f}")

    def _is_critical_experiment(self, dataset, regime, method):
        """Определяем, нужно ли сохранять эмбеддинги."""
        critical = [
            ('cifar10', 'scarcity_05', 'hems'),
            ('cifar10', 'scarcity_05', 'baseline'),
            ('cifar100', 'longtail_10', 'hems'),
            ('cifar100', 'longtail_10', 'baseline'),
        ]
        return (dataset, regime, method) in critical

    def save_embeddings(self, train_emb, test_emb, train_labels, test_labels,
                        dataset, regime, method, seed):
        """Save embeddings for later analysis."""
        save_dir = f"./embeddings/{dataset}/{regime}/{method}/seed_{seed}"
        os.makedirs(save_dir, exist_ok=True)

        np.savez_compressed(
            os.path.join(save_dir, 'embeddings.npz'),
            train_embeddings=train_emb,
            test_embeddings=test_emb,
            train_labels=train_labels,
            test_labels=test_labels
        )
        print(f"  💾 Embeddings saved to {save_dir}")

    def run_all(self):
        """Run all experiments with optimized settings."""
        print("=" * 60)
        print("OPTIMIZED HEMS EXPERIMENTS")
        print("=" * 60)
        print(f"Device: {self.device}")
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CIFAR-10 Seeds: {self.config.seeds_cifar10}")
        print(f"CIFAR-100 Seeds: {self.config.seeds_cifar100}")
        print("=" * 60)
        print("Epochs by regime:")
        for key, epochs in self.config.epochs_by_regime.items():
            print(f"  {key}: {epochs}")
        print("=" * 60)

        # Определяем эксперименты
        experiments = [
            # CIFAR-10: ВСЕ режимы, но с динамическими эпохами
            {
                'dataset': 'cifar10',
                'regime': 'balanced_100',
                'fraction': 1.0,
                'imbalance': None,
                'seeds': self.config.seeds_cifar10,
                'num_classes': 10,
                'loader': self.cifar10_loader,
                'epochs': self.config.get_epochs('cifar10', 'balanced_100')
            },
            {
                'dataset': 'cifar10',
                'regime': 'scarcity_30',
                'fraction': 0.3,
                'imbalance': None,
                'seeds': self.config.seeds_cifar10,
                'num_classes': 10,
                'loader': self.cifar10_loader,
                'epochs': self.config.get_epochs('cifar10', 'scarcity_30')
            },
            {
                'dataset': 'cifar10',
                'regime': 'scarcity_10',
                'fraction': 0.1,
                'imbalance': None,
                'seeds': self.config.seeds_cifar10,
                'num_classes': 10,
                'loader': self.cifar10_loader,
                'epochs': self.config.get_epochs('cifar10', 'scarcity_10')
            },
            {
                'dataset': 'cifar10',
                'regime': 'scarcity_05',
                'fraction': 0.05,
                'imbalance': None,
                'seeds': self.config.seeds_cifar10,
                'num_classes': 10,
                'loader': self.cifar10_loader,
                'epochs': self.config.get_epochs('cifar10', 'scarcity_05')
            },
            {
                'dataset': 'cifar10',
                'regime': 'longtail_10',
                'fraction': 1.0,
                'imbalance': 10,
                'seeds': self.config.seeds_cifar10,
                'num_classes': 10,
                'loader': self.cifar10_loader,
                'epochs': self.config.get_epochs('cifar10', 'longtail_10')
            },
            {
                'dataset': 'cifar10',
                'regime': 'longtail_50',
                'fraction': 1.0,
                'imbalance': 50,
                'seeds': self.config.seeds_cifar10,
                'num_classes': 10,
                'loader': self.cifar10_loader,
                'epochs': self.config.get_epochs('cifar10', 'longtail_50')
            },
            # CIFAR-100: ВСЕ режимы, но меньше seeds
            {
                'dataset': 'cifar100',
                'regime': 'balanced_100',
                'fraction': 1.0,
                'imbalance': None,
                'seeds': self.config.seeds_cifar100,
                'num_classes': 100,
                'loader': self.cifar100_loader,
                'epochs': self.config.get_epochs('cifar100', 'balanced_100')
            },
            {
                'dataset': 'cifar100',
                'regime': 'scarcity_10',
                'fraction': 0.1,
                'imbalance': None,
                'seeds': self.config.seeds_cifar100,
                'num_classes': 100,
                'loader': self.cifar100_loader,
                'epochs': self.config.get_epochs('cifar100', 'scarcity_10')
            },
            {
                'dataset': 'cifar100',
                'regime': 'longtail_10',
                'fraction': 1.0,
                'imbalance': 10,
                'seeds': self.config.seeds_cifar100,
                'num_classes': 100,
                'loader': self.cifar100_loader,
                'epochs': self.config.get_epochs('cifar100', 'longtail_10')
            },
        ]

        # Подсчет общего количества экспериментов
        self.total = sum(len(exp['seeds']) * 2 for exp in experiments)
        print(f"\nTotal experiments: {self.total}")
        print(f"Estimated time: ~{self.total * 15 / 60:.1f} hours (with RTX 5080)")
        print("=" * 60)

        # Запуск
        for exp in experiments:
            self.run_experiment(
                exp['dataset'],
                exp['regime'],
                exp['fraction'],
                exp['imbalance'],
                exp['seeds'],
                exp['num_classes'],
                exp['loader'],
                exp['epochs']
            )

            # Save checkpoint after each regime
            self.save_checkpoint(f"regime_{exp['dataset']}_{exp['regime']}")

        # Final save
        df = self.save_final_results()

        # Print summary
        elapsed = time.time() - self.start_time
        print(f"\n{'=' * 60}")
        print(f"ALL EXPERIMENTS COMPLETED!")
        print(f"Total time: {elapsed / 3600:.1f} hours")
        print(f"Experiments: {self.completed}")
        print(f"{'=' * 60}")

        return df


if __name__ == "__main__":
    # Create config
    config = Config()

    # Run experiments
    runner = OptimizedExperimentRunner(config)
    results_df = runner.run_all()

    # Print summary
    if results_df is not None:
        print("\n" + "=" * 60)
        print("Summary of Results")
        print("=" * 60)
        print(results_df.groupby(['dataset', 'regime', 'method'])[['val_acc', 'val_f1']].mean())