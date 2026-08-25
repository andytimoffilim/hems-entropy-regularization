# HEMS: Hyperspherical Entropy Maximization with Class Separation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.10+](https://img.shields.io/badge/pytorch-1.10+-red.svg)](https://pytorch.org/)

This repository contains the official implementation of the paper:

**"Generalization of Entropy-Regularized Embeddings: Distributional Spreading, Sample Coverage, and Empirical Evaluation of HEMS"**

Authors: Andrey V. Timofeev, Alexandr S. Anufriev

---

## 📄 Abstract

This paper studies entropy regularization of embedding distributions and its application to Hyperspherical Entropy Maximization with Class Separation (HEMS). We provide a theoretical analysis of three distinct concepts: proximity of an embedding distribution to a uniform reference distribution, coverage of the high-probability region of the embedding space, and uniform geometric coverage of the full support. 

The empirical evaluation on CIFAR-10 and CIFAR-100 reveals a non-trivial relationship between geometric properties and predictive performance. HEMS consistently improves geometric metrics (effective rank +957%, coverage +613%) but the impact on classification accuracy is task-dependent.

---

## 📦 Requirements

- Python 3.8+
- PyTorch 1.10+
- CUDA (optional, for GPU acceleration)

Install dependencies:

```bash
pip install -r requirements.txt
🚀 Quick Start
1. Clone the repository
bash
git clone https://github.com/andytimoffilim/HEMS.git
cd HEMS
2. Run experiments
bash
python main_experiment.py
This will:

Download CIFAR-10 and CIFAR-100 automatically

Train Baseline and HEMS models across all stress regimes

Compute geometric metrics (effective rank, R_hat, quantization entropy)

Save results to results/

3. Generate visualizations
bash
python visualize_results.py
This will produce:

Scarcity curves (Macro-F1 vs training fraction)

Coverage vs performance scatter plots

Geometry comparison box plots

Correlation heatmaps

📁 Repository Structure
text
HEMS/
├── config.py                 # Configuration and hyperparameters
├── data_loader.py            # CIFAR-10/100 with stress regimes
├── models.py                 # ResNet-18 architecture
├── losses.py                 # HEMS loss components
├── train.py                  # Training loop
├── evaluate.py               # Geometric metrics computation
├── main_experiment.py        # Run all experiments
├── visualize_results.py      # Generate plots and tables
├── requirements.txt          # Dependencies
├── results/                  # CSV files with results
│   └── results_final_*.csv   # Final results used in the paper
└── README.md
🧪 Experiments
The code reproduces the following experiments:

Dataset	Regime	Description
CIFAR-10	Balanced 100%	Full training set
CIFAR-10	Scarcity 30%, 10%, 5%	Label scarcity
CIFAR-10	Longtail IR=10, 50	Class imbalance
CIFAR-100	Balanced 100%	Full training set
CIFAR-100	Scarcity 10%	Label scarcity
CIFAR-100	Longtail IR=10	Class imbalance
Each experiment runs 3 seeds for CIFAR-10 and 2 seeds for CIFAR-100.

📊 Results
The final results are available in results/results_final_20260824_164325.csv.

Key findings:

HEMS consistently improves geometric metrics (effective rank +957%, coverage +613%)

On CIFAR-100 Longtail, HEMS improves accuracy by 2.74% and Macro-F1 by 4.64% (p=0.02)

On CIFAR-10 Scarcity 5%, HEMS degrades accuracy by 2.85% (p=0.002)

This reveals the Geometry-Performance Paradox: improved geometry does not guarantee better classification

🖼️ Visualizations
Run visualize_results.py to generate:

Figure	Description
scarcity_curves_cifar10.png	Macro-F1 and R_hat vs training fraction
scarcity_curves_cifar100.png	Same for CIFAR-100
coverage_vs_performance_cifar10.png	R_hat vs Macro-F1 scatter plot
coverage_vs_performance_cifar100.png	Same for CIFAR-100
geometry_comparison_cifar10.png	Box plots of geometric metrics
correlation_heatmap_cifar10.png	Correlation matrix of all metrics
📝 Citation
If you use this code in your research, please cite our paper:

bibtex
@article{timofeev2026hems,
  title={Generalization of Entropy-Regularized Embeddings: Distributional Spreading, Sample Coverage, and Empirical Evaluation of HEMS},
  author={Timofeev, Andrey V. and Anufriev, Alexandr S.},
  journal={},
  year={2026},
  volume={},
  pages={}
}
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgements
We thank the AI Center for SCO+ Countries for providing computational resources.

📧 Contact
Andrey V. Timofeev: tav@centeraisco.com

