<div align="center">

# Nature-Inspired Feature Selection for Intrusion Detection

**A controlled comparison of Genetic Algorithm, Particle Swarm Optimization, and Ant Colony Optimization as wrapper-based feature selectors for network intrusion detection**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](notebooks/nature_inspired_feature_selection.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/nature-inspired-ids-feature-selection/blob/main/notebooks/nature_inspired_feature_selection.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Paper (PDF)](docs/paper.pdf) · [Slides (PDF)](docs/presentation.pdf) · [Notebook](notebooks/nature_inspired_feature_selection.ipynb) · [Results](results/)

</div>

---

## Overview

Machine-learning intrusion detection systems (IDS) work on network-traffic records described by
dozens to hundreds of features, many of them redundant or irrelevant. Removing them cuts
training and inference cost and can improve detection quality.

GA, PSO, and ACO are all popular wrapper feature selectors, but published studies usually test
**one algorithm on one dataset** with their own preprocessing, classifier, and metrics, so the
results cannot be compared. This project fixes that by running all three under an
**identical experimental protocol**:

| Held constant | Setting |
|---|---|
| Datasets | NSL-KDD and CIC-IDS-2017 (binary: normal vs. attack) |
| Preprocessing | Median/mode imputation, one-hot encoding, min–max scaling |
| Classifier | Random Forest (100 trees for final evaluation) |
| Fitness | `Fitness(C) = w1 · F1(C) − w2 · |C| / n`, with `w1 = 0.9`, `w2 = 0.1` |
| Evaluation | Accuracy, Precision, Detection Rate (Recall), F1, FPR, feature reduction, runtime |
| Statistics | Wilcoxon rank-sum (pairwise) and Friedman (all three) tests over independent runs |

So any difference in the results comes from the **search strategy itself**.

> Course project for **MAI 650 – Nature-Inspired Computing**, Canadian University Dubai.

## Key Results

Results from the pilot-scale run (`FAST_MODE = True`), evaluated on the **full held-out test set**.

### NSL-KDD (122 features after one-hot encoding)

| Method | Features | Reduction | Accuracy | Detection Rate | F1 | FPR |
|---|---:|---:|---:|---:|---:|---:|
| Baseline RF (all features) | 122 | 0.0% | 0.7649 | 0.6078 | 0.7464 | 0.0274 |
| GA  | 43 | 64.8% | 0.7644 | 0.6077 | 0.7460 | 0.0285 |
| **PSO** | 52 | 57.4% | **0.8194** | **0.7026** | **0.8158** | **0.0263** |
| ACO | **25** | **79.5%** | 0.7727 | 0.6224 | 0.7571 | 0.0286 |

### CIC-IDS-2017 (78 features)

| Method | Features | Reduction | Accuracy | Detection Rate | F1 | FPR |
|---|---:|---:|---:|---:|---:|---:|
| Baseline RF (all features) | 78 | 0.0% | 0.9981 | 0.9927 | 0.9952 | 0.0006 |
| **GA** | 29 | 62.8% | **0.9990** | **0.9964** | **0.9974** | **0.0004** |
| PSO | 44 | 43.6% | 0.9982 | 0.9929 | 0.9953 | 0.0005 |
| ACO | **16** | **79.5%** | 0.9985 | 0.9948 | 0.9963 | 0.0005 |

### Takeaways

- **Feature selection keeps or improves detection quality** while removing **43–80 %** of the features.
- **ACO** gives the **smallest subsets** on both datasets (about 80 % reduction).
- **PSO** gives the **best detection rate** on the harder NSL-KDD benchmark (+9.5 points over the baseline).
- **GA** gives the best overall scores on CIC-IDS-2017.
- Wilcoxon and Friedman tests found **no statistically significant difference** among the three
  algorithms at the pilot sample size (Friedman p = 0.25 on NSL-KDD, p = 0.55 on CIC-IDS-2017).
  So the right choice depends on what you need most: the smallest model, the highest detection
  rate, or the lowest cost.

### Figures

| Convergence | Algorithm comparison |
|---|---|
| ![Convergence curves](results/figures/convergence_curves.png) | ![Algorithm comparison](results/figures/algorithm_comparison.png) |

![Confusion matrices](results/figures/confusion_matrices.png)

## Methodology

```
 Raw data ──► Preprocessing ──► Wrapper search (GA | PSO | ACO) ──► Best feature mask ──► Final RF ──► Metrics
                                      │                ▲
                                      ▼                │
                              Random Forest + k-fold CV fitness
```

| Algorithm | Encoding | Main operators |
|---|---|---|
| **GA** | Binary chromosome (1 = keep feature) | Roulette-wheel selection, single-point crossover (pc = 0.8), bit-flip mutation (pm = 0.02), best-so-far tracking |
| **PSO** | Continuous position + sigmoid binarisation | Inertia weight decaying 0.9 → 0.4, c1 = c2 = 2.0, threshold θ = 0.5 |
| **ACO** | Node-based pheromone per feature | α = 1.0, β = 2.0, evaporation ρ = 0.1, subset size limited to 20–80 % of features |

All three are **implemented from scratch** (no metaheuristic libraries), following the pseudocode
in Section III of the [paper](docs/paper.pdf). Each one stops early when the best fitness has not
improved for `patience` generations.

## Repository Structure

```
nature-inspired-ids-feature-selection/
├── notebooks/
│   └── nature_inspired_feature_selection.ipynb   # full pipeline: load → search → evaluate → plot → stats
├── data/
│   └── README.md                                 # how to download and place the datasets
├── results/
│   ├── main_results.csv                          # metrics for each dataset × method
│   └── figures/                                  # convergence, comparison, confusion matrices
├── docs/
│   ├── paper.pdf                                 # full research paper
│   └── presentation.pdf                          # presentation slides
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/nature-inspired-ids-feature-selection.git
cd nature-inspired-ids-feature-selection
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get the datasets

The datasets are **not included** because they are too large for GitHub (about 1 GB). Follow
[data/README.md](data/README.md) to download them into `data/`.

### 3. Run

```bash
jupyter notebook notebooks/nature_inspired_feature_selection.ipynb
```

Then run all cells. The notebook finds out by itself whether it is running on **Google Colab**
(it mounts Drive) or **locally** (it uses the repository root). Outputs are written to `results/`.

### Configuration

The **CONFIGURATION** cell controls runtime:

| Setting | `FAST_MODE = True` (default) | `FAST_MODE = False` (paper settings) |
|---|---|---|
| Population / swarm / colony | 12 | 50 |
| Generations / iterations | 8 | 100 |
| CV folds in fitness | 3 | 5 |
| RF trees during search | 40 | 100 |
| Rows used for fitness | 6,000 subsample | full training set |
| Independent runs for statistics | 5 | 30 |
| Approx. runtime (CPU) | a few minutes | several hours |

In both modes, the final selected subset is evaluated on the **full** test set with a
100-tree Random Forest, so the reported metrics can be trusted in either mode.

## Reproducibility

Every algorithm and the fitness function take an explicit `random_state` (default `42`).
The statistical-validation cell uses seeds `100, 101, …`, so each run is independent and
can still be reproduced exactly.

## Citation

If you use this work, please cite it (see also [`CITATION.cff`](CITATION.cff)):

```bibtex
@misc{nour2026natureinspired,
  author = {Nour, Najma Abdulkarim},
  title  = {Nature-Inspired Feature Selection for Intrusion Detection Systems:
            A Comparative Study of Genetic Algorithm, Particle Swarm Optimization,
            and Ant Colony Optimization},
  year   = {2026},
  note   = {MAI 650 -- Nature-Inspired Computing, Canadian University Dubai},
  howpublished = {\url{https://github.com/<your-username>/nature-inspired-ids-feature-selection}}
}
```

## Acknowledgements

- **NSL-KDD** and **CIC-IDS-2017** datasets: [Canadian Institute for Cybersecurity, University of New Brunswick](https://www.unb.ca/cic/datasets/index.html).
- Built with NumPy, pandas, scikit-learn, SciPy, and Matplotlib.

## License

The code is released under the [MIT License](LICENSE). The datasets are subject to their own
terms from the Canadian Institute for Cybersecurity.
