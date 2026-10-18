"""Command-line entry point: ``nifs run`` and ``nifs stats``."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from nifs import __version__
from nifs.algorithms import ALGORITHMS
from nifs.config import ExperimentConfig
from nifs.data import DATASET_NAMES, load_dataset
from nifs.evaluation import evaluate_subset
from nifs.fitness import make_fitness_fn
from nifs.plotting import plot_comparison, plot_confusion_matrices, plot_convergence
from nifs.stats import compare_algorithms

log = logging.getLogger("nifs")


def _search(cfg: ExperimentConfig, algo: str, X, y, seed: int):
    fitness_fn, n_features = make_fitness_fn(
        X, y, cv_folds=cfg.cv_folds, n_trees=cfg.rf_trees_search,
        subsample_rows=cfg.subsample_rows, w1=cfg.w1, w2=cfg.w2, random_state=seed,
    )
    return ALGORITHMS[algo](fitness_fn, n_features, pop_size=cfg.pop_size, n_gen=cfg.n_gen,
                            patience=cfg.patience, random_state=seed)


def _load_all(cfg: ExperimentConfig) -> dict:
    datasets = {}
    for key in cfg.datasets:
        log.info("Loading %s ...", DATASET_NAMES.get(key, key))
        datasets[DATASET_NAMES.get(key, key)] = load_dataset(
            key, cfg.data_dir, cic_row_cap=cfg.cic_row_cap, random_state=cfg.random_state)
    return datasets


def run_experiment(cfg: ExperimentConfig) -> pd.DataFrame:
    """Baseline + every algorithm on every dataset; writes CSVs and figures."""
    out = cfg.output_dir
    fig_dir = os.path.join(out, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    rows, histories, cms, selected = [], {}, {}, {}
    for ds_name, (X_tr, X_te, y_tr, y_te, feat_names) in _load_all(cfg).items():
        log.info("=== %s: %d features, %d train / %d test rows ===",
                 ds_name, X_tr.shape[1], len(y_tr), len(y_te))

        t0 = time.time()
        full = np.ones(X_tr.shape[1], dtype=bool)
        m, _, y_pred = evaluate_subset(full, X_tr, y_tr, X_te, y_te, n_trees=cfg.rf_trees_final,
                                       random_state=cfg.random_state, return_model=True)
        rows.append(dict(dataset=ds_name, algorithm="Baseline", **m,
                         runtime_sec=round(time.time() - t0, 2)))
        cms[(ds_name, "Baseline")] = confusion_matrix(y_te, y_pred, labels=[0, 1])

        for algo in cfg.algorithms:
            t0 = time.time()
            res = _search(cfg, algo, X_tr, y_tr, cfg.random_state)
            elapsed = round(time.time() - t0, 2)
            m, _, y_pred = evaluate_subset(res.mask, X_tr, y_tr, X_te, y_te,
                                           n_trees=cfg.rf_trees_final,
                                           random_state=cfg.random_state, return_model=True)
            rows.append(dict(dataset=ds_name, algorithm=algo, **m, runtime_sec=elapsed,
                             search_fitness=res.fitness, n_generations=len(res.history) - 1))
            histories[(ds_name, algo)] = res.history
            cms[(ds_name, algo)] = confusion_matrix(y_te, y_pred, labels=[0, 1])
            selected[f"{ds_name}/{algo}"] = [f for f, k in zip(feat_names, res.mask) if k]
            log.info("  %-4s k=%d/%d  F1=%.4f  DR=%.4f  FPR=%.4f  time=%.1fs", algo,
                     m["n_selected"], m["n_total"], m["f1"], m["recall_DR"], m["fpr"], elapsed)

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(out, "main_results.csv"), index=False, float_format="%.4f")
    with open(os.path.join(out, "selected_features.json"), "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2)
    with open(os.path.join(out, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    plot_convergence(histories, os.path.join(fig_dir, "convergence_curves.png"))
    plot_comparison(results[results.algorithm != "Baseline"],
                    os.path.join(fig_dir, "algorithm_comparison.png"), cfg.algorithms)
    plot_confusion_matrices(cms, os.path.join(fig_dir, "confusion_matrices.png"))
    log.info("Results written to %s", out)
    return results


def run_statistics(cfg: ExperimentConfig) -> pd.DataFrame:
    """Repeat each search ``n_stat_runs`` times (seeds 100, 101, ...) and test significance."""
    records = []
    for ds_name, (X_tr, X_te, y_tr, y_te, _) in _load_all(cfg).items():
        for algo in cfg.algorithms:
            for run in range(cfg.n_stat_runs):
                seed = 100 + run
                res = _search(cfg, algo, X_tr, y_tr, seed)
                m = evaluate_subset(res.mask, X_tr, y_tr, X_te, y_te,
                                    n_trees=cfg.rf_trees_final, random_state=seed)
                records.append(dict(dataset=ds_name, algorithm=algo, run=run, **m))
            log.info("  done: %s / %s", ds_name, algo)

    os.makedirs(cfg.output_dir, exist_ok=True)
    runs = pd.DataFrame(records)
    runs.to_csv(os.path.join(cfg.output_dir, "statistical_runs.csv"), index=False)
    tests = compare_algorithms(runs)
    tests.to_csv(os.path.join(cfg.output_dir, "statistical_tests.csv"), index=False)
    log.info("\n%s", tests.to_string(index=False))
    return tests


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="nifs", description="GA / PSO / ACO wrapper feature selection for IDS.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [("run", "run baseline + all algorithms, save results and figures"),
                            ("stats", "repeated runs + Wilcoxon / Friedman tests")]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("-c", "--config", default="configs/fast.yaml", help="YAML config file")
        p.add_argument("--datasets", nargs="+", choices=sorted(DATASET_NAMES))
        p.add_argument("--algorithms", nargs="+", choices=list(ALGORITHMS))
        p.add_argument("--data-dir")
        p.add_argument("--output-dir")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    cfg = ExperimentConfig.from_yaml(args.config)
    for key in ("datasets", "algorithms", "data_dir", "output_dir"):
        if getattr(args, key) is not None:
            setattr(cfg, key, getattr(args, key))

    if args.command == "run":
        run_experiment(cfg)
    else:
        run_statistics(cfg)


if __name__ == "__main__":
    main()
