"""Final evaluation of a feature subset on the held-out test set."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_subset(mask, X_train, y_train, X_test, y_test, n_trees: int = 100,
                    random_state: int = 42, return_model: bool = False):
    """Train a Random Forest on the selected features and score it on the test set.

    Returns a metrics dict (accuracy, precision, recall_DR, f1, fpr, subset size,
    feature reduction %). With ``return_model=True`` also returns
    ``(clf, y_pred)``.
    """
    mask = np.asarray(mask, dtype=bool)
    clf = RandomForestClassifier(n_estimators=n_trees, n_jobs=-1, random_state=random_state)
    clf.fit(X_train[:, mask], y_train)
    y_pred = clf.predict(X_test[:, mask])

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    metrics = dict(
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, zero_division=0),
        recall_DR=recall_score(y_test, y_pred, zero_division=0),   # Detection Rate
        f1=f1_score(y_test, y_pred, zero_division=0),
        fpr=fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        n_selected=int(mask.sum()),
        n_total=len(mask),
        feature_reduction_pct=100 * (1 - mask.sum() / len(mask)),
    )
    if return_model:
        return metrics, clf, y_pred
    return metrics
