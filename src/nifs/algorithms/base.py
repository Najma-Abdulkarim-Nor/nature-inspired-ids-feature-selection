from __future__ import annotations

from typing import NamedTuple

import numpy as np


class SearchResult(NamedTuple):
    """Output of a feature-selection search."""

    mask: np.ndarray          # boolean mask over features
    fitness: float            # best fitness found
    history: list[float]      # best-so-far fitness per generation / iteration
