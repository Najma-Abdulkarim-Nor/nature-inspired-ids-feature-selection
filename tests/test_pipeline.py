import numpy as np
import pandas as pd
import pytest

from nifs.config import ExperimentConfig
from nifs.data import preprocess
from nifs.evaluation import evaluate_subset
from nifs.stats import compare_algorithms


def test_fitness_penalises_subset_size(toy_fitness):
    fitness_fn, n = toy_fitness
    assert fitness_fn(np.zeros(n)) == -1.0
    # with w1 = 0.9, the best possible fitness is 0.9 minus the size penalty
    assert fitness_fn(np.ones(n)) <= 0.9 - 0.1


def test_preprocess_scales_and_one_hot_encodes():
    train = pd.DataFrame({"num": [0.0, 5.0, 10.0, np.nan], "proto": ["tcp", "udp", "tcp", "icmp"],
                          "label": [0, 1, 0, 1]})
    test = pd.DataFrame({"num": [20.0, -5.0], "proto": ["udp", "unseen"], "label": [1, 0]})

    X_tr, X_te, y_tr, y_te, names = preprocess(train, test)

    assert names == ["num", "proto_icmp", "proto_tcp", "proto_udp"]
    assert X_tr.min() >= 0 and X_tr.max() <= 1
    assert X_te.min() >= 0 and X_te.max() <= 1          # test clipped to train range
    assert X_tr[3, 0] == pytest.approx(0.5)              # NaN imputed with train median (5)
    np.testing.assert_array_equal(X_te[1, 1:], [0, 0, 0])  # unseen category -> all zeros
    np.testing.assert_array_equal(y_te, [1, 0])


def test_evaluate_subset_reports_reduction(toy_data):
    X, y = toy_data
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[:5] = True
    m = evaluate_subset(mask, X[:200], y[:200], X[200:], y[200:], n_trees=10)

    assert m["n_selected"] == 5 and m["n_total"] == 20
    assert m["feature_reduction_pct"] == pytest.approx(75.0)
    for key in ("accuracy", "precision", "recall_DR", "f1", "fpr"):
        assert 0.0 <= m[key] <= 1.0


def test_compare_algorithms_runs_both_tests():
    rng = np.random.RandomState(0)
    df = pd.DataFrame([dict(dataset="D", algorithm=a, run=r, f1=rng.rand())
                       for a in ("GA", "PSO", "ACO") for r in range(6)])
    out = compare_algorithms(df)

    assert set(out.test) == {"Wilcoxon signed-rank", "Friedman"}
    assert len(out) == 4


def test_config_files_load():
    for path in ("configs/fast.yaml", "configs/paper.yaml"):
        cfg = ExperimentConfig.from_yaml(path)
        assert cfg.algorithms == ["GA", "PSO", "ACO"]
