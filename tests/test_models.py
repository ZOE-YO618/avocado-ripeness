"""Tests for the baseline, kNN helpers and the logistic ripening model."""

import numpy as np
import pytest

from avocado_ripeness.baseline import ThresholdClassifier, hue_ripeness_score
from avocado_ripeness.classifier import kiosk_message, make_knn, select_k
from avocado_ripeness.ripening_model import (days_until_ready, fit_logistic, logistic_bounded,
                                             q10, time_to_index)


# ----------------------------------------------------------------- baseline
def test_hue_unwrapping_is_monotonic_with_ripening():
    # green (90) > yellow (40) > red (5) > purple (330 -> -30)
    s = hue_ripeness_score([90, 40, 5, 330])
    assert np.all(np.diff(s) < 0)


def test_threshold_classifier_recovers_perfectly_separable_classes():
    rng = np.random.default_rng(1)
    centres = {1: 90, 2: 60, 3: 30, 4: 0, 5: -30}
    y = np.repeat([1, 2, 3, 4, 5], 40)
    x = np.array([centres[c] for c in y]) + rng.uniform(-8, 8, y.size)
    clf = ThresholdClassifier().fit(x, y)
    assert np.mean(clf.predict(x) == y) == 1.0


def test_threshold_classifier_requires_fit():
    with pytest.raises(RuntimeError):
        ThresholdClassifier().predict([1.0])


# ----------------------------------------------------------------- kNN
def test_knn_and_k_selection_on_toy_data():
    rng = np.random.default_rng(0)
    y = np.repeat([1, 2, 3], 30)
    X = rng.normal(size=(90, 2)) * 0.3 + y[:, None]
    groups = np.arange(90) // 3
    k, scores = select_k(X, y, groups, k_values=(1, 3, 5))
    assert k in (1, 3, 5) and len(scores) == 3
    assert make_knn(k).fit(X, y).score(X, y) > 0.9


def test_kiosk_messages():
    assert "ready to eat" in kiosk_message(3, 0.9, 0.55, 5.5)
    assert "not recommended" in kiosk_message(5, 0.9, 0.55, 5.5)
    assert "more day" in kiosk_message(1, 0.9, 0.55, 5.5)


# ----------------------------------------------------------------- ripening model
def test_logistic_fit_recovers_known_parameters():
    """Verification against a manufactured solution with known k and t0."""
    t = np.repeat(np.arange(1, 21), 5).astype(float)
    rng = np.random.default_rng(3)
    r = logistic_bounded(t, k=0.6, t0=8.0) + rng.normal(0, 0.05, t.size)
    fit = fit_logistic(t, r, "bounded")
    assert fit.params["k"] == pytest.approx(0.6, rel=0.05)
    assert fit.params["t0"] == pytest.approx(8.0, abs=0.2)
    assert fit.r2 > 0.98


def test_three_parameter_fit_runs():
    t = np.linspace(1, 20, 60)
    fit = fit_logistic(t, 5 / (1 + np.exp(-0.5 * (t - 8))), "3p")
    assert fit.params["r_max"] == pytest.approx(5, rel=1e-3)
    assert fit.rmse < 1e-6


def test_fit_rejects_too_little_data():
    with pytest.raises(ValueError):
        fit_logistic([1, 2], [1, 2])


def test_time_to_index_inverts_the_curve():
    k, t0 = 0.5, 6.0
    for target in (1.5, 3.0, 4.5):
        t = time_to_index(target, k, t0)
        assert logistic_bounded(t, k, t0) == pytest.approx(target)
    with pytest.raises(ValueError):
        time_to_index(5.0, k, t0)


def test_days_until_ready():
    assert days_until_ready(3, 0.5, 6.0) == 0.0
    assert days_until_ready(1, 0.5, 6.0) > days_until_ready(2, 0.5, 6.0) > 0


def test_q10():
    assert q10(0.25, 0.5) == pytest.approx(2.0)
