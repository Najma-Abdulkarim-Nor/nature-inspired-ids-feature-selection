import numpy as np
import pytest

from nifs.algorithms import ALGORITHMS


@pytest.mark.parametrize("name", list(ALGORITHMS))
def test_search_returns_valid_mask(name, toy_fitness):
    fitness_fn, n = toy_fitness
    res = ALGORITHMS[name](fitness_fn, n, pop_size=4, n_gen=3, random_state=0)

    assert res.mask.dtype == bool
    assert res.mask.shape == (n,)
    assert 0 < res.mask.sum() <= n
    assert np.isfinite(res.fitness)


@pytest.mark.parametrize("name", list(ALGORITHMS))
def test_history_is_monotone_best_so_far(name, toy_fitness):
    fitness_fn, n = toy_fitness
    res = ALGORITHMS[name](fitness_fn, n, pop_size=4, n_gen=3, random_state=0)

    assert len(res.history) >= 1
    assert all(b >= a for a, b in zip(res.history, res.history[1:]))
    assert res.history[-1] == pytest.approx(res.fitness)


@pytest.mark.parametrize("name", list(ALGORITHMS))
def test_search_is_reproducible(name, toy_fitness):
    fitness_fn, n = toy_fitness
    a = ALGORITHMS[name](fitness_fn, n, pop_size=4, n_gen=2, random_state=7)
    b = ALGORITHMS[name](fitness_fn, n, pop_size=4, n_gen=2, random_state=7)

    np.testing.assert_array_equal(a.mask, b.mask)
    assert a.fitness == b.fitness


def test_aco_respects_subset_size_bounds(toy_fitness):
    fitness_fn, n = toy_fitness
    res = ALGORITHMS["ACO"](fitness_fn, n, pop_size=4, n_gen=2,
                            min_size_frac=0.2, max_size_frac=0.5, random_state=0)
    assert int(0.2 * n) <= res.mask.sum() <= int(0.5 * n)
