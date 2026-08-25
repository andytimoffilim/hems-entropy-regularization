import os
from dataclasses import dataclass
from typing import Tuple, List, Dict
import torch


@dataclass
class Config:
    # Data paths
    data_root: str = "E:/datasets"

    # Model parameters
    embedding_dim: int = 128
    num_classes_cifar10: int = 10
    num_classes_cifar100: int = 100

    # Training parameters
    batch_size: int = 128
    learning_rate: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9

    # HEMS parameters - УСИЛЕННЫЕ
    temperature: float = 0.1
    lambda_u_max: float = 0.1  # Было 0.05
    lambda_d: float = 0.05  # Было 0.01
    lambda_s: float = 0.1  # Было 0.02
    warmup_epochs: int = 20
    ema_momentum: float = 0.99

    # Quantization parameters
    num_centroids: int = 256
    eta_values: List[float] = (0.05, 0.1)
    a_values: List[float] = (0.001, 0.005, 0.01)

    # -------- НОВАЯ СТРАТЕГИЯ ЭПОХ --------
    epochs_by_regime: Dict[str, int] = None

    # -------- НОВАЯ СТРАТЕГИЯ SEEDS --------
    # CIFAR-10: 3 seeds для статистической значимости
    # CIFAR-100: 2 seeds (экономия)
    seeds_cifar10: List[int] = (42, 123, 456)
    seeds_cifar100: List[int] = (42, 123)

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # CIFAR-10-C path
    cifar10_c_path: str = None

    def __post_init__(self):
        """Инициализация динамических эпох."""
        if self.epochs_by_regime is None:
            self.epochs_by_regime = {
                # CIFAR-10: критически важные режимы - 200 эпох
                'cifar10_balanced_100': 200,  # Критично: проверка, что не ломает
                'cifar10_scarcity_30': 150,  # Меньше данных → быстрее сходится
                'cifar10_scarcity_10': 120,  # Еще меньше данных
                'cifar10_scarcity_05': 200,  # Критично: экстремальный случай
                'cifar10_longtail_10': 200,  # Критично: дисбаланс
                'cifar10_longtail_50': 200,  # Критично: сильный дисбаланс

                # CIFAR-100: 200 эпох для всех (сложный датасет)
                'cifar100_balanced_100': 200,  # Критично
                'cifar100_scarcity_10': 150,  # Мало данных
                'cifar100_longtail_10': 200,  # Критично: где HEMS показал +4.5%
            }

    def get_epochs(self, dataset: str, regime: str) -> int:
        """Получить количество эпох для конкретного эксперимента."""
        key = f"{dataset}_{regime}"
        return self.epochs_by_regime.get(key, 150)  # По умолчанию 150