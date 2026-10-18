"""Figures for the results section: convergence, comparison, confusion matrices."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

METRIC_LABELS = {
    "f1": "F1-score",
    "recall_DR": "Detection Rate",
    "fpr": "False Positive Rate",
    "feature_reduction_pct": "Feature Reduction (%)",
    "runtime_sec": "Search Runtime (s)",
}


def plot_convergence(histories: dict[tuple[str, str], list[float]], path: str) -> None:
    """``histories`` maps (dataset, algorithm) to best-so-far fitness per generation."""
    datasets = list(dict.fromkeys(ds for ds, _ in histories))
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.5), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        for (d, algo), hist in histories.items():
            if d == ds:
                ax.plot(range(1, len(hist) + 1), hist, marker="o", markersize=3, label=algo)
        ax.set_title(f"Convergence — {ds}")
        ax.set_xlabel("Generation / Iteration")
        ax.set_ylabel("Best-so-far fitness")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison(results: pd.DataFrame, path: str, algorithms: list[str]) -> None:
    """Grouped bars of each metric, one group per algorithm, one bar per dataset."""
    datasets = list(dict.fromkeys(results["dataset"]))
    metrics = list(METRIC_LABELS)
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.5))
    width = 0.8 / len(datasets)
    for ax, metric in zip(axes, metrics):
        for d_idx, ds in enumerate(datasets):
            sub = results[results.dataset == ds].set_index("algorithm")
            vals = [sub.loc[a, metric] for a in algorithms]
            ax.bar(np.arange(len(algorithms)) + d_idx * width, vals, width=width, label=ds)
        ax.set_xticks(np.arange(len(algorithms)) + width * (len(datasets) - 1) / 2)
        ax.set_xticklabels(algorithms)
        ax.set_title(METRIC_LABELS[metric])
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrices(cms: dict[tuple[str, str], np.ndarray], path: str) -> None:
    """``cms`` maps (dataset, method) to a 2x2 confusion matrix."""
    datasets = list(dict.fromkeys(ds for ds, _ in cms))
    methods = list(dict.fromkeys(m for _, m in cms))
    labels = ["Normal", "Attack"]
    fig, axes = plt.subplots(len(datasets), len(methods),
                             figsize=(3.6 * len(methods), 3.6 * len(datasets)), squeeze=False)
    for r, ds in enumerate(datasets):
        for c, method in enumerate(methods):
            ax = axes[r, c]
            cm = cms[(ds, method)]
            ax.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black")
            ax.set_xticks([0, 1], labels)
            ax.set_yticks([0, 1], labels)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"{ds}\n{method}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
