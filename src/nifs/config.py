"""Experiment configuration, loadable from a YAML file."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class ExperimentConfig:
    """All settings for one experiment run.

    Defaults match ``configs/fast.yaml``; ``configs/paper.yaml`` holds the
    full-scale settings reported in the paper (Tables II-IV).
    """

    datasets: list[str] = field(default_factory=lambda: ["nsl-kdd", "cic-ids-2017"])
    algorithms: list[str] = field(default_factory=lambda: ["GA", "PSO", "ACO"])
    data_dir: str = "data"
    output_dir: str = "results"

    # search budget
    pop_size: int = 12          # population / swarm / colony size
    n_gen: int = 8              # generations / iterations
    patience: int = 6           # early stopping: generations without improvement

    # fitness: w1 * F1 - w2 * |C| / n, estimated with k-fold CV
    cv_folds: int = 3
    rf_trees_search: int = 40
    subsample_rows: int | None = 6000
    w1: float = 0.9
    w2: float = 0.1

    # final evaluation and statistics
    rf_trees_final: int = 100
    n_stat_runs: int = 5
    cic_row_cap: int | None = 150_000
    random_state: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        return cls(**raw)

    def to_dict(self) -> dict:
        return asdict(self)
