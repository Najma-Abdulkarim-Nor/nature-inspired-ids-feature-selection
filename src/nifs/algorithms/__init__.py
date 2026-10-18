"""Binary wrapper feature-selection metaheuristics."""

from nifs.algorithms.aco import run_aco
from nifs.algorithms.base import SearchResult
from nifs.algorithms.ga import run_ga
from nifs.algorithms.pso import run_pso

ALGORITHMS = {"GA": run_ga, "PSO": run_pso, "ACO": run_aco}

__all__ = ["ALGORITHMS", "SearchResult", "run_aco", "run_ga", "run_pso"]
