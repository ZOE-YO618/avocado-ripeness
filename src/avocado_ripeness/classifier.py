"""
k-Nearest Neighbours ripeness classifier and kiosk-style prediction helper.

kNN predicts the class of a new avocado by finding the ``k`` training
avocados whose colour feature vectors are closest (Euclidean distance) and
taking a majority vote. Features are standardised first (zero mean, unit
variance) because otherwise features with large numeric ranges (hue in
degrees, 0-360) would dominate features measured on a 0-1 scale.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .ripening_model import days_until_ready

STAGE_NAMES = {
    1: "Underripe",
    2: "Breaking",
    3: "Ripe (first phase)",
    4: "Ripe (second phase)",
    5: "Overripe",
}


def make_knn(k: int = 5, weights: str = "distance"):
    """Return an (unfitted) StandardScaler + kNN pipeline.

    ``weights="distance"`` gives closer neighbours a larger vote.
    """
    return make_pipeline(StandardScaler(),
                         KNeighborsClassifier(n_neighbors=k, weights=weights))


def select_k(X, y, groups, k_values=(1, 3, 5, 7, 9, 11, 15, 21), n_splits=5):
    """Choose k by grouped cross-validation on the training data only.

    Groups are avocado IDs, so photos of the same fruit never appear in both
    the fitting and validation parts of a fold.

    Returns
    -------
    best_k : int
    scores : dict
        Mean macro-F1 for every k tried.
    """
    n_splits = min(n_splits, len(np.unique(groups)))
    cv = GroupKFold(n_splits=n_splits)
    scores = {}
    for k in k_values:
        s = cross_val_score(make_knn(k), X, y, groups=groups, cv=cv, scoring="f1_macro")
        scores[k] = float(s.mean())
    best_k = max(scores, key=scores.get)
    return best_k, scores


def kiosk_message(stage: int, confidence: float, k_rate: float, t0: float) -> str:
    """Turn a predicted stage into the message a shopper would see.

    Parameters
    ----------
    stage : int
        Predicted ripening stage (1-5).
    confidence : float
        Classifier probability for that stage (0-1).
    k_rate, t0 : float
        Fitted logistic parameters for room-temperature storage, used to
        estimate days until ready.
    """
    name = STAGE_NAMES[stage]
    conf = f"(confidence {confidence:.0%})"
    if stage in (3, 4):
        return f"{name}: ready to eat now {conf}"
    if stage == 5:
        return f"{name}: past best quality, not recommended {conf}"
    days = days_until_ready(stage, k_rate, t0)
    return f"{name}: about {days:.0f} more day(s) at room temperature {conf}"
