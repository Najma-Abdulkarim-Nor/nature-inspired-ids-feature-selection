"""Ant Colony Optimization feature selection (paper Algorithm 3)."""

from __future__ import annotations

import numpy as np

from nifs.algorithms.base import SearchResult
from nifs.fitness import FitnessFn


def run_aco(fitness_fn: FitnessFn, n_features: int, pop_size: int = 12, n_gen: int = 8,
            rho: float = 0.1, alpha: float = 1.0, beta: float = 2.0,
            min_size_frac: float = 0.2, max_size_frac: float = 0.8, patience: int = 6,
            random_state: int = 42) -> SearchResult:
    """Node-based ACO: each ant builds a subset by sampling features with
    probability proportional to ``tau^alpha * eta^beta``.

    ``eta`` is a single-feature fitness heuristic computed once on a sample of
    features; ``pop_size`` is the number of ants.
    """
    rng = np.random.RandomState(random_state)
    tau = np.full(n_features, 1.0 / n_features)

    eta = np.zeros(n_features)
    sample_idx = rng.choice(n_features, size=min(n_features, max(10, n_features // 4)),
                            replace=False)
    for j in sample_idx:
        mask = np.zeros(n_features, dtype=int)
        mask[j] = 1
        eta[j] = max(fitness_fn(mask), 1e-6)
    eta[eta == 0] = eta[sample_idx].mean() if len(sample_idx) else 1.0

    min_k = max(1, int(min_size_frac * n_features))
    max_k = max(min_k + 1, int(max_size_frac * n_features))

    best_mask, best_fit = None, -np.inf
    history = []
    no_improve, it = 0, 0

    while it < n_gen and no_improve < patience:
        iter_best_fit, iter_best_mask = -np.inf, None
        subset_masks = []

        for _ in range(pop_size):
            allowed = list(range(n_features))
            chosen = []
            target_size = rng.randint(min_k, max_k + 1)
            while len(chosen) < target_size and allowed:
                probs = tau[allowed] ** alpha * eta[allowed] ** beta
                probs = probs / probs.sum()
                pick = rng.choice(allowed, p=probs)
                chosen.append(pick)
                allowed.remove(pick)

            mask = np.zeros(n_features, dtype=int)
            mask[chosen] = 1
            f = fitness_fn(mask)
            subset_masks.append((mask, f))
            if f > iter_best_fit:
                iter_best_fit, iter_best_mask = f, mask

        deposit = np.zeros(n_features)
        for mask, f in subset_masks:
            deposit += mask * max(f, 0)
        tau = (1 - rho) * tau + deposit

        if iter_best_fit > best_fit:
            best_fit, best_mask, no_improve = iter_best_fit, iter_best_mask, 0
        else:
            no_improve += 1
        history.append(best_fit)
        it += 1

    return SearchResult(best_mask.astype(bool), best_fit, history)
