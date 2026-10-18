"""Significance tests over repeated independent runs (paper Section IV-F)."""

from __future__ import annotations

from itertools import combinations

import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


def compare_algorithms(stat_df: pd.DataFrame, metric: str = "f1") -> pd.DataFrame:
    """Run paired Wilcoxon signed-rank tests and a Friedman test per dataset.

    ``stat_df`` needs columns ``dataset``, ``algorithm``, ``run`` and ``metric``.
    Runs with the same index share a seed, so the samples are paired.
    """
    rows = []
    for ds_name, sub in stat_df.groupby("dataset", sort=False):
        pivot = sub.pivot(index="run", columns="algorithm", values=metric)
        for a, b in combinations(pivot.columns, 2):
            try:
                stat, p = wilcoxon(pivot[a], pivot[b])
            except ValueError:   # e.g. all differences are zero
                stat, p = float("nan"), float("nan")
            rows.append(dict(dataset=ds_name, test="Wilcoxon signed-rank",
                             comparison=f"{a} vs {b}", statistic=stat, p_value=p))
        if pivot.shape[1] >= 3:
            stat, p = friedmanchisquare(*[pivot[c].values for c in pivot.columns])
            rows.append(dict(dataset=ds_name, test="Friedman",
                             comparison=" vs ".join(pivot.columns), statistic=stat, p_value=p))
    return pd.DataFrame(rows)
