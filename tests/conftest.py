import numpy as np
import pytest
from sklearn.datasets import make_classification


@pytest.fixture(scope="session")
def toy_data():
    """Small binary problem: 6 informative features among 20."""
    X, y = make_classification(n_samples=300, n_features=20, n_informative=6,
                               n_redundant=2, random_state=0)
    return X.astype(np.float32), y


@pytest.fixture(scope="session")
def toy_fitness(toy_data):
    from nifs.fitness import make_fitness_fn

    X, y = toy_data
    return make_fitness_fn(X, y, cv_folds=2, n_trees=5, subsample_rows=None)
