import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

from config import Config
from data_loader import CIFAR10DataLoader, CIFAR100DataLoader
from models import CIFARResNet18
from train import Trainer
from evaluate import EmbeddingAnalyzer
from losses import BaselineLoss


def load_imagenet_encoder(device='cuda'):
    """Загружает предобученный ResNet-18 (ImageNet) и возвращает энкодер (без fc)."""
    import torchvision.models as models
    print("Загрузка предобученного ResNet-18 (ImageNet)...")
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet.fc = nn.Identity()  # убираем классификатор
    resnet = resnet.to(device)
    for param in resnet.parameters():
        param.requires_grad = False
    print("✅ Энкодер загружен и заморожен.")
    return resnet


def create_model(encoder, num_classes, embedding_dim=128, device='cuda'):
    """Создаёт CIFARResNet18 с подменённым backbone."""
    model = CIFARResNet18(embedding_dim=embedding_dim, num_classes=num_classes)
    model.resnet = encoder
    for param in model.resnet.parameters():
        param.requires_grad = False
    model.to(device)
    return model


def run_experiment(config, dataset_name, regime_name, fraction, imbalance_ratio,
                   seeds, num_classes, dataloader, encoder, epochs=100,
                   results_list=None, csv_path=None):
    """
    Запускает эксперименты для данного режима.
    results_list — глобальный список для накопления результатов.
    csv_path — путь для сохранения после каждого seed.
    """
    print(f"\n=== IMAGENET: {dataset_name} - {regime_name} (epochs={epochs}) ===")

    train_set, test_set = dataloader.load_full_data()
    train_subset, val_subset = dataloader.create_validation_split(train_set)

    for seed in seeds:
        print(f"\n  Seed: {seed}")

        # Создаём подвыборку
        if 'scarcity' in regime_name:
            train_dataset = dataloader.create_label_scarcity_split(train_subset, fraction, seed)
        elif 'longtail' in regime_name:
            train_dataset = dataloader.create_long_tailed_split(train_subset, imbalance_ratio, seed)
        else:
            train_dataset = train_subset

        dataloaders = dataloader.get_dataloaders(train_dataset, val_subset, test_set)

        # Создаём модель с новым классификатором для каждого seed
        model = create_model(encoder, num_classes, config.embedding_dim, config.device)

        # Настройка тренера
        save_dir = f"./checkpoints_imagenet/{dataset_name}/{regime_name}/seed_{seed}"
        trainer_config = {
            'device': config.device,
            'learning_rate': 0.01,
            'momentum': config.momentum,
            'weight_decay': config.weight_decay,
            'epochs': epochs
        }
        loss_fn = BaselineLoss()
        trainer = Trainer(model, loss_fn, trainer_config, save_dir)

        print(f"    Обучение классификатора ({epochs} эпох)...")
        start = time.time()
        trainer.train(dataloaders['train'], dataloaders['val'], epochs)
        train_time = time.time() - start

        # Загружаем лучшую модель
        best_path = trainer.get_best_checkpoint_path()
        if best_path:
            trainer.load_checkpoint(best_path)

        # Валидация
        _, val_acc, val_f1 = trainer.validate(dataloaders['val'])

        # Геометрические метрики
        analyzer = EmbeddingAnalyzer(config.embedding_dim, config.num_centroids)
        train_emb, train_lbl = analyzer.extract_embeddings(model, dataloaders['train'], config.device)
        test_emb, test_lbl = analyzer.extract_embeddings(model, dataloaders['test'], config.device)
        geo_metrics = analyzer.compute_all_metrics(train_emb, test_emb, train_lbl, test_lbl)

        result = {
            'dataset': dataset_name,
            'regime': regime_name,
            'method': 'imagenet',
            'seed': seed,
            'epochs': epochs,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'train_time_min': train_time / 60,
            **geo_metrics
        }

        # Сохраняем результат
        results_list.append(result)
        if csv_path:
            pd.DataFrame(results_list).to_csv(csv_path, index=False)
            print(f"    💾 Результат сохранён в {csv_path}")

        print(f"    Результат: acc={val_acc:.4f}, f1={val_f1:.4f}, время={train_time / 60:.1f} мин")


def main():
    config = Config()

    # Эксперименты (только критические)
    experiments = [
        {
            'dataset': 'cifar10',
            'regime': 'scarcity_05',
            'fraction': 0.05,
            'imbalance': None,
            'seeds': config.seeds_cifar10,
            'num_classes': 10,
            'loader': CIFAR10DataLoader(config.data_root, config.batch_size),
            'epochs': 100
        },
        {
            'dataset': 'cifar10',
            'regime': 'longtail_10',
            'fraction': 1.0,
            'imbalance': 10,
            'seeds': config.seeds_cifar10,
            'num_classes': 10,
            'loader': CIFAR10DataLoader(config.data_root, config.batch_size),
            'epochs': 100
        },
        {
            'dataset': 'cifar100',
            'regime': 'longtail_10',
            'fraction': 1.0,
            'imbalance': 10,
            'seeds': config.seeds_cifar100,
            'num_classes': 100,
            'loader': CIFAR100DataLoader(config.data_root, config.batch_size),
            'epochs': 100
        }
    ]

    # Загружаем энкодер ОДИН РАЗ
    encoder = load_imagenet_encoder(config.device)

    # Глобальный список результатов
    all_results = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f'results_imagenet_{timestamp}.csv'

    for exp in experiments:
        run_experiment(
            config,
            exp['dataset'],
            exp['regime'],
            exp['fraction'],
            exp['imbalance'],
            exp['seeds'],
            exp['num_classes'],
            exp['loader'],
            encoder,
            epochs=exp['epochs'],
            results_list=all_results,
            csv_path=csv_path
        )

    print(f"\n✅ Все результаты сохранены в {csv_path}")
    df = pd.DataFrame(all_results)
    print("\n=== Сводка средних значений ===")
    print(df.groupby(['dataset', 'regime'])[['val_acc', 'val_f1']].mean().round(4))


if __name__ == "__main__":
    main()