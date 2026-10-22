"""
Numerical model of ripening progression over time.

The ripening index R (1 = underripe ... 5 = overripe) of an avocado stored
under constant conditions is modelled with a logistic (sigmoid) curve of
time t in days:

Proposed form (3 parameters, from the project proposal)::

    R(t) = R_max / (1 + exp(-k (t - t0)))

Bounded form (2 parameters, used as the final model)::

    R(t) = R_min + (R_max - R_min) / (1 + exp(-k (t - t0)))
    with R_min = 1 and R_max = 5 fixed by the definition of the index.

k  [1/day] is the ripening rate constant (steepness of the curve) and
t0 [day]   is the time of the inflection point (mid-way through ripening).

Parameters are found by nonlinear least squares (Levenberg-Marquardt via
``scipy.optimize.curve_fit``), which minimises the sum of squared residuals
between the curve and every observed (t, R) point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

R_MIN, R_MAX = 1.0, 5.0


def logistic_3p(t, r_max, k, t0):
    """Three-parameter logistic curve R(t) = r_max / (1 + exp(-k (t - t0)))."""
    return r_max / (1.0 + np.exp(-k * (np.asarray(t, float) - t0)))


def logistic_bounded(t, k, t0, r_min=R_MIN, r_max=R_MAX):
    """Logistic curve bounded between ``r_min`` and ``r_max``."""
    return r_min + (r_max - r_min) / (1.0 + np.exp(-k * (np.asarray(t, float) - t0)))


@dataclass
class FitResult:
    """Fitted parameters and goodness-of-fit for one storage condition."""

    model: str
    params: dict
    std_errors: dict
    rmse: float
    r2: float
    n: int

    def predict(self, t):
        """Evaluate the fitted curve at times ``t`` (days)."""
        if self.model == "3p":
            return logistic_3p(t, **self.params)
        return logistic_bounded(t, **self.params)


def _metrics(y, yhat):
    resid = np.asarray(y, float) - np.asarray(yhat, float)
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else float("nan")
    return rmse, r2


def fit_logistic(t, r, model: str = "bounded") -> FitResult:
    """Fit a logistic ripening curve to observations.

    Parameters
    ----------
    t : array_like
        Day of experiment for each observation (days, >= 1).
    r : array_like
        Ripening index for each observation (1-5).
    model : {"bounded", "3p"}
        Which curve to fit (see module docstring).

    Returns
    -------
    FitResult
        Fitted parameters, their standard errors (square root of the diagonal
        of the covariance matrix returned by ``curve_fit``), RMSE and R^2.

    Raises
    ------
    ValueError
        If fewer than 3 observations are given or the lengths differ.
    """
    t = np.asarray(t, float)
    r = np.asarray(r, float)
    if t.shape != r.shape or t.size < 3:
        raise ValueError("Need at least 3 paired (t, R) observations.")

    t_mid = float(np.median(t))
    if model == "3p":
        p0 = [R_MAX, 0.5, t_mid]
        popt, pcov = curve_fit(logistic_3p, t, r, p0=p0, maxfev=20000)
        names = ["r_max", "k", "t0"]
        yhat = logistic_3p(t, *popt)
    elif model == "bounded":
        p0 = [0.5, t_mid]
        popt, pcov = curve_fit(lambda tt, k, t0: logistic_bounded(tt, k, t0),
                               t, r, p0=p0, maxfev=20000)
        names = ["k", "t0"]
        yhat = logistic_bounded(t, *popt)
    else:
        raise ValueError(f"Unknown model '{model}'")

    perr = np.sqrt(np.diag(pcov))
    rmse, r2 = _metrics(r, yhat)
    return FitResult(model, dict(zip(names, map(float, popt))),
                     dict(zip(names, map(float, perr))), rmse, r2, int(t.size))


def time_to_index(target: float, k: float, t0: float,
                  r_min: float = R_MIN, r_max: float = R_MAX) -> float:
    """Invert the bounded logistic: time at which R(t) reaches ``target``.

    Solving ``target = r_min + (r_max - r_min) / (1 + exp(-k (t - t0)))``
    for t gives ``t = t0 - ln((r_max - r_min) / (target - r_min) - 1) / k``.
    """
    if not (r_min < target < r_max):
        raise ValueError("target must lie strictly between r_min and r_max")
    return t0 - np.log((r_max - r_min) / (target - r_min) - 1.0) / k


def days_until_ready(stage: int, k: float, t0: float, ready_index: float = 3.0) -> float:
    """Estimate days until a fruit at ``stage`` reaches the ready-to-eat stage.

    The fruit's current position on the curve is taken as the time at which
    the curve passes the middle of its stage (stage 1 -> 1.5, stage 2 -> 2.5).
    Returns 0 for stages that are already ready (>= ``ready_index``).
    """
    if stage >= ready_index:
        return 0.0
    t_now = time_to_index(stage + 0.5, k, t0)
    t_ready = time_to_index(ready_index, k, t0)
    return max(0.0, float(t_ready - t_now))


def q10(k_low: float, k_high: float, t_low: float = 10.0, t_high: float = 20.0) -> float:
    """Temperature coefficient Q10 = (k_high / k_low) ** (10 / (T_high - T_low)).

    Q10 is the factor by which the ripening rate increases for a 10 degC rise,
    a standard way to express temperature sensitivity in postharvest science.
    """
    return float((k_high / k_low) ** (10.0 / (t_high - t_low)))
