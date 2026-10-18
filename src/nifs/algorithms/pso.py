"""Binary Particle Swarm Optimization feature selection (paper Algorithm 2)."""

from __future__ import annotations

import numpy as np

from nifs.algorithms.base import SearchResult
from nifs.fitness import FitnessFn


def run_pso(fitness_fn: FitnessFn, n_features: int, pop_size: int = 12, n_gen: int = 8,
            w_start: float = 0.9, w_end: float = 0.4, c1: float = 2.0, c2: float = 2.0,
            theta: float = 0.5, patience: int = 6, random_state: int = 42) -> SearchResult:
    """PSO over continuous positions in [0, 1], binarised with a sigmoid threshold.

    The inertia weight decays linearly from ``w_start`` to ``w_end``.
    """
    rng = np.random.RandomState(random_state)

    def binarize(x):
        return (1 / (1 + np.exp(-x)) > theta).astype(int)

    X = rng.rand(pop_size, n_features)
    V = rng.uniform(-1, 1, size=(pop_size, n_features))
    pbest = X.copy()
    pbest_fit = np.array([fitness_fn(binarize(X[i])) for i in range(pop_size)])
    gbest_idx = int(np.argmax(pbest_fit))
    gbest, gbest_fit = pbest[gbest_idx].copy(), pbest_fit[gbest_idx]
    history = [gbest_fit]

    no_improve, t = 0, 0
    while t < n_gen and no_improve < patience:
        w = w_start - (w_start - w_end) * (t / max(n_gen - 1, 1))
        r1 = rng.rand(pop_size, n_features)
        r2 = rng.rand(pop_size, n_features)
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        X = np.clip(X + V, 0, 1)

        for i in range(pop_size):
            f = fitness_fn(binarize(X[i]))
            if f > pbest_fit[i]:
                pbest[i], pbest_fit[i] = X[i].copy(), f

        cur_idx = int(np.argmax(pbest_fit))
        if pbest_fit[cur_idx] > gbest_fit:
            gbest, gbest_fit, no_improve = pbest[cur_idx].copy(), pbest_fit[cur_idx], 0
        else:
            no_improve += 1
        history.append(gbest_fit)
        t += 1

    return SearchResult(binarize(gbest).astype(bool), gbest_fit, history)
