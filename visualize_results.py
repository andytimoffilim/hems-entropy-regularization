import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, ttest_rel
import warnings

warnings.filterwarnings('ignore')

# Настройка стиля
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12


class ResultVisualizer:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.datasets = self.df['dataset'].unique()

    def plot_scarcity_curves(self, dataset='cifar10', save=True):
        """Plot Macro-F1 vs training fraction for scarcity experiments."""
        df_scarcity = self.df[
            (self.df['dataset'] == dataset) &
            (self.df['regime'].str.contains('scarcity'))
            ].copy()

        # Извлекаем fraction из названия режима
        df_scarcity['fraction'] = df_scarcity['regime'].str.extract('(\d+\.?\d*)').astype(float) / 100

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Macro-F1
        for method in df_scarcity['method'].unique():
            data = df_scarcity[df_scarcity['method'] == method]
            mean = data.groupby('fraction')['val_f1'].mean()
            std = data.groupby('fraction')['val_f1'].std()

            axes[0].plot(mean.index, mean.values,
                         label=method.upper(), marker='o', linewidth=2, markersize=8)
            axes[0].fill_between(mean.index, mean - std, mean + std, alpha=0.2)

        axes[0].set_xlabel('Training Fraction')
        axes[0].set_ylabel('Macro-F1')
        axes[0].set_title(f'{dataset.upper()}: Macro-F1 vs Training Data')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: R_hat (coverage)
        for method in df_scarcity['method'].unique():
            data = df_scarcity[df_scarcity['method'] == method]
            mean = data.groupby('fraction')['R_hat'].mean()
            std = data.groupby('fraction')['R_hat'].std()

            axes[1].plot(mean.index, mean.values,
                         label=method.upper(), marker='s', linewidth=2, markersize=8)
            axes[1].fill_between(mean.index, mean - std, mean + std, alpha=0.2)

        axes[1].set_xlabel('Training Fraction')
        axes[1].set_ylabel('Coverage Proxy R_hat')
        axes[1].set_title(f'{dataset.upper()}: Coverage vs Training Data')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save:
            plt.savefig(f'scarcity_curves_{dataset}.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_coverage_vs_performance(self, dataset='cifar10', save=True):
        """Scatter plot: R_hat vs Macro-F1."""
        df_filtered = self.df[self.df['dataset'] == dataset]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: All regimes
        for method in df_filtered['method'].unique():
            data = df_filtered[df_filtered['method'] == method]
            axes[0].scatter(data['R_hat'], data['val_f1'],
                            label=method.upper(), alpha=0.7, s=80)

        # Корреляция
        corr, p_value = spearmanr(df_filtered['R_hat'], df_filtered['val_f1'])
        axes[0].set_xlabel('Coverage Proxy R_hat')
        axes[0].set_ylabel('Macro-F1')
        axes[0].set_title(f'{dataset.upper()}: All Regimes\nSpearman ρ = {corr:.3f} (p={p_value:.4f})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: По режимам
        regimes = df_filtered['regime'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(regimes)))

        for i, regime in enumerate(regimes):
            data = df_filtered[df_filtered['regime'] == regime]
            axes[1].scatter(data['R_hat'], data['val_f1'],
                            label=regime, alpha=0.7, s=80, color=colors[i])

        axes[1].set_xlabel('Coverage Proxy R_hat')
        axes[1].set_ylabel('Macro-F1')
        axes[1].set_title(f'{dataset.upper()}: Colored by Regime')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save:
            plt.savefig(f'coverage_vs_performance_{dataset}.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_geometry_comparison(self, dataset='cifar10', save=True):
        """Compare geometry metrics between Baseline and HEMS."""
        df_filtered = self.df[self.df['dataset'] == dataset]

        metrics = ['effective_rank', 'R_hat', 'mean_pairwise_sim', 'inter_intra_ratio']
        titles = ['Effective Rank', 'Coverage Proxy (R_hat)',
                  'Mean Pairwise Similarity', 'Inter/Intra Ratio']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx]

            # Box plots
            data_to_plot = []
            labels = []
            for method in ['baseline', 'hems']:
                data = df_filtered[df_filtered['method'] == method][metric]
                data_to_plot.append(data)
                labels.append(method.upper())

            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)

            # Цвета
            colors = ['lightblue', 'lightgreen']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)

            ax.set_ylabel(title)
            ax.set_title(f'{dataset.upper()}: {title}')
            ax.grid(True, alpha=0.3)

            # Добавляем статистику
            for i, data in enumerate(data_to_plot):
                mean_val = np.mean(data)
                ax.text(i + 1, mean_val, f'μ={mean_val:.3f}',
                        ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        if save:
            plt.savefig(f'geometry_comparison_{dataset}.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_class_performance(self, dataset='cifar10', regime='longtail_50', save=True):
        """Plot class-wise performance for long-tail experiments."""
        df_filtered = self.df[
            (self.df['dataset'] == dataset) &
            (self.df['regime'] == regime)
            ]

        # Для class-wise нужно больше данных - пока пропускаем
        print(f"Class-wise analysis for {dataset}/{regime} requires additional data")

    def plot_heatmap(self, dataset='cifar10', save=True):
        """Plot correlation heatmap of all metrics."""
        df_filtered = self.df[self.df['dataset'] == dataset]

        # Выбираем числовые колонки
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['seed']]

        # Корреляционная матрица
        corr_matrix = df_filtered[numeric_cols].corr()

        plt.figure(figsize=(14, 12))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, linewidths=0.5,
                    cbar_kws={'shrink': 0.8})
        plt.title(f'{dataset.upper()}: Correlation Matrix of All Metrics')
        plt.tight_layout()
        if save:
            plt.savefig(f'correlation_heatmap_{dataset}.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_summary_table(self, dataset='cifar10'):
        """Generate summary table for LaTeX."""
        df_filtered = self.df[self.df['dataset'] == dataset]

        summary = df_filtered.groupby(['regime', 'method']).agg({
            'val_acc': ['mean', 'std'],
            'val_f1': ['mean', 'std'],
            'R_hat': ['mean', 'std'],
            'effective_rank': ['mean', 'std']
        }).round(4)

        print(f"\n{'=' * 60}")
        print(f"SUMMARY TABLE: {dataset.upper()}")
        print(f"{'=' * 60}")
        print(summary)

        return summary

    def statistical_tests(self, dataset='cifar10'):
        """Perform paired t-tests between Baseline and HEMS."""
        df_filtered = self.df[self.df['dataset'] == dataset]

        results = []
        for regime in df_filtered['regime'].unique():
            data = df_filtered[df_filtered['regime'] == regime]

            baseline = data[data['method'] == 'baseline']['val_f1'].values
            hems = data[data['method'] == 'hems']['val_f1'].values

            if len(baseline) == len(hems) and len(baseline) > 1:
                t_stat, p_value = ttest_rel(baseline, hems)
                diff = np.mean(hems - baseline)

                results.append({
                    'regime': regime,
                    'diff': diff,
                    't_stat': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })

        df_results = pd.DataFrame(results)
        print(f"\n{'=' * 60}")
        print(f"STATISTICAL TESTS: {dataset.upper()}")
        print(f"{'=' * 60}")
        print(df_results.to_string())

        return df_results

    def run_all_analysis(self):
        """Run all visualizations."""
        for dataset in self.datasets:
            print(f"\n{'#' * 60}")
            print(f"ANALYZING: {dataset.upper()}")
            print(f"{'#' * 60}")

            # 1. Scarcity curves
            self.plot_scarcity_curves(dataset, save=True)

            # 2. Coverage vs performance
            self.plot_coverage_vs_performance(dataset, save=True)

            # 3. Geometry comparison
            self.plot_geometry_comparison(dataset, save=True)

            # 4. Heatmap
            self.plot_heatmap(dataset, save=True)

            # 5. Summary table
            self.generate_summary_table(dataset)

            # 6. Statistical tests
            self.statistical_tests(dataset)


if __name__ == "__main__":
    # Используйте ваш файл с результатами
    visualizer = ResultVisualizer('results_20260821_122033.csv')
    visualizer.run_all_analysis()

    # Дополнительно: отдельные графики
    print("\n" + "=" * 60)
    print("INDIVIDUAL PLOTS FOR CIFAR-10 AND CIFAR-100")
    print("=" * 60)

    # Только для CIFAR-10
    visualizer.plot_scarcity_curves('cifar10', save=True)
    visualizer.plot_coverage_vs_performance('cifar10', save=True)

    # Только для CIFAR-100
    visualizer.plot_scarcity_curves('cifar100', save=True)
    visualizer.plot_coverage_vs_performance('cifar100', save=True)