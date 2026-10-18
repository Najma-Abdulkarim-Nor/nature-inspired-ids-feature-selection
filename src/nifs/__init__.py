"""Nature-inspired feature selection (GA, PSO, ACO) for intrusion detection."""

from nifs.algorithms import ALGORITHMS, SearchResult, run_aco, run_ga, run_pso
from nifs.config import ExperimentConfig
from nifs.data import load_dataset, preprocess
from nifs.evaluation import evaluate_subset
from nifs.fitness import make_fitness_fn

__version__ = "1.0.0"

__all__ = [
    "ALGORITHMS",
    "ExperimentConfig",
    "SearchResult",
    "evaluate_subset",
    "load_dataset",
    "make_fitness_fn",
    "preprocess",
    "run_aco",
    "run_ga",
    "run_pso",
]
