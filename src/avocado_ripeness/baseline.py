"""
Baseline: a simple colour-threshold rule.

The baseline uses ONE scalar colour feature (e.g. mean hue angle) and four
cut-points that split its range into the five ripening stages. The cut-points
are chosen from the training data by a brute-force search that maximises
training accuracy, assuming a monotonic relationship between the feature and
ripeness (as the skin goes green -> olive -> purple -> dark purple).

This mimics what a human or a simple colour sensor would do ("if the skin is
darker than X, call it ripe") and gives an honest reference point that the
kNN classifier must beat to justify its extra complexity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def hue_ripeness_score(hue_mean_deg: np.ndarray) -> np.ndarray:
    """Map mean hue angle to a monotonic 'ripeness direction' score.

    Green skin has hue around 60-100 deg; as the fruit ripens the hue moves
    down through yellow/orange (~30 deg) and wraps past 0 deg into purple/red
    (~330-360 deg). Unwrapping the angle so that purple values become small
    numbers makes the score decrease monotonically with ripening:
    ``score = hue`` for hue <= 180, and ``hue - 360`` for hue > 180.

    Parameters
    ----------
    hue_mean_deg : array_like
        Circular mean hue angle(s) in degrees, range [0, 360).

    Returns
    -------
    numpy.ndarray
        Unwrapped hue in degrees, range (-180, 180]; larger = greener.
    """
    h = np.asarray(hue_mean_deg, dtype=float)
    return np.where(h > 180.0, h - 360.0, h)


@dataclass
class ThresholdClassifier:
    """Ordinal threshold classifier on a single scalar feature.

    Parameters
    ----------
    classes : sequence of int
        Ordered class labels, e.g. ``(1, 2, 3, 4, 5)``.
    decreasing : bool
        True if the feature value *decreases* as the class index increases
        (true for unwrapped hue: green = high, purple = low).
    n_candidates : int
        Number of candidate cut-point positions (quantiles of the training
        feature) considered in the greedy search.
    """

    classes: tuple = (1, 2, 3, 4, 5)
    decreasing: bool = True
    n_candidates: int = 60
    thresholds_: np.ndarray = field(default=None, init=False)

    def _assign(self, x: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        xs = -x if self.decreasing else x
        idx = np.searchsorted(np.sort(thresholds), xs, side="right")
        return np.asarray(self.classes)[idx]

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ThresholdClassifier":
        """Choose cut-points that maximise training accuracy.

        Uses coordinate ascent: start from cut-points placed at the class
        boundaries implied by the class proportions, then repeatedly move
        each cut-point to the candidate value that gives the highest
        accuracy while keeping the cut-points ordered.
        """
        x = np.asarray(x, float)
        y = np.asarray(y)
        xs = -x if self.decreasing else x
        cands = np.unique(np.quantile(xs, np.linspace(0, 1, self.n_candidates)))

        # initial guess from cumulative class proportions
        props = np.cumsum([np.mean(y == c) for c in self.classes])[:-1]
        thr = np.quantile(xs, props)

        best = np.mean(self._assign(x, thr) == y)
        for _ in range(5):  # a few sweeps are enough to converge
            improved = False
            for j in range(len(thr)):
                lo = thr[j - 1] if j > 0 else -np.inf
                hi = thr[j + 1] if j < len(thr) - 1 else np.inf
                for c in cands[(cands > lo) & (cands < hi)]:
                    trial = thr.copy()
                    trial[j] = c
                    acc = np.mean(self._assign(x, trial) == y)
                    if acc > best:
                        best, thr, improved = acc, trial, True
            if not improved:
                break
        self.thresholds_ = thr
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict class labels for feature values ``x``."""
        if self.thresholds_ is None:
            raise RuntimeError("Call fit() before predict().")
        return self._assign(np.asarray(x, float), self.thresholds_)
