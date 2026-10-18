"""Genetic Algorithm feature selection (paper Algorithm 1)."""

from __future__ import annotations

import numpy as np

from nifs.algorithms.base import SearchResult
from nifs.fitness import FitnessFn


def run_ga(fitness_fn: FitnessFn, n_features: int, pop_size: int = 12, n_gen: int = 8,
           pc: float = 0.8, pm: float = 0.02, patience: int = 6,
           random_state: int = 42) -> SearchResult:
    """Binary GA with roulette-wheel selection, single-point crossover and bit-flip mutation."""
    rng = np.random.RandomState(random_state)

    def roulette_select(pop, fits):
        fits = np.array(fits)
        fits = fits - fits.min() + 1e-6
        probs = fits / fits.sum()
        return pop[rng.choice(len(pop), p=probs)]

    def crossover(p1, p2):
        m = rng.randint(1, n_features)
        return np.concatenate([p1[:m], p2[m:]]), np.concatenate([p2[:m], p1[m:]])

    def mutate(c):
        flip = rng.rand(n_features) < pm
        c = c.copy()
        c[flip] = 1 - c[flip]
        return c

    pop = [(rng.rand(n_features) < 0.5).astype(int) for _ in range(pop_size)]
    fits = [fitness_fn(c) for c in pop]
    best = pop[int(np.argmax(fits))].copy()
    best_fit = max(fits)
    history = [best_fit]
    no_improve, g = 0, 0

    while g < n_gen and no_improve < patience:
        new_pop = []
        while len(new_pop) < pop_size:
            p1, p2 = roulette_select(pop, fits), roulette_select(pop, fits)
            c1, c2 = crossover(p1, p2) if rng.rand() < pc else (p1.copy(), p2.copy())
            new_pop.append(mutate(c1))
            if len(new_pop) < pop_size:
                new_pop.append(mutate(c2))
        pop = new_pop
        fits = [fitness_fn(c) for c in pop]
        cur_idx = int(np.argmax(fits))
        if fits[cur_idx] > best_fit:
            best, best_fit, no_improve = pop[cur_idx].copy(), fits[cur_idx], 0
        else:
            no_improve += 1
        history.append(best_fit)
        g += 1

    return SearchResult(best.astype(bool), best_fit, history)
