"""Shared wrapper fitness: Fitness(C) = w1 * F1(C) - w2 * |C| / n  (paper Eq. 2)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

FitnessFn = Callable[[np.ndarray], float]


def make_fitness_fn(X, y, cv_folds: int = 3, n_trees: int = 40,
                    subsample_rows: int | None = 6000, w1: float = 0.9, w2: float = 0.1,
                    random_state: int = 42) -> tuple[FitnessFn, int]:
    """Build the fitness function shared by all three algorithms.

    F1 is the mean k-fold cross-validated F1 of a Random Forest trained on the
    candidate subset, optionally on a fixed row subsample for speed. An empty
    subset scores -1.
    """
    n_features = X.shape[1]
    rng = np.random.RandomState(random_state)

    if subsample_rows is not None and subsample_rows < X.shape[0]:
        idx = rng.choice(X.shape[0], size=subsample_rows, replace=False)
        Xs, ys = X[idx], y[idx]
    else:
        Xs, ys = X, y

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    def fitness(mask) -> float:
        mask = np.asarray(mask).astype(bool)
        k = mask.sum()
        if k == 0:
            return -1.0
        clf = RandomForestClassifier(n_estimators=n_trees, n_jobs=-1, random_state=random_state)
        f1 = cross_val_score(clf, Xs[:, mask], ys, cv=skf, scoring="f1", n_jobs=1).mean()
        return w1 * f1 - w2 * (k / n_features)

    return fitness, n_features
