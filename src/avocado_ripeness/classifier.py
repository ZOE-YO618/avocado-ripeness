"""
Ripeness prediction models and the kiosk-style message.

The ripening index (1 = underripe ... 5 = overripe) is an *ordered* scale, so
the main approach treats it as a **regression** target: a model predicts a
continuous index (e.g. 3.4), which is then rounded to the nearest stage.
Compared with plain classification this means "off by one stage" is a smaller
error than "off by three", which matches how wrong a recommendation would be.

Models compared (all trained on the same standardised colour features):

=================  =============================================  ==============
name               model                                          tuned settings
=================  =============================================  ==============
``linear``         ordinary least-squares linear regression       none
``tree``           decision tree regression                       depth, leaf size
``svr``            support vector regression, RBF kernel          C, epsilon, gamma
``mlp``            small neural network (multi-layer perceptron)  layers, L2 penalty
``knn``            k-nearest-neighbours *classifier*              k
=================  =============================================  ==============

Linear regression is the final model; the others are evidence-based
alternatives (see the report).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .ripening_model import days_until_ready

STAGE_NAMES = {
    1: "Underripe",
    2: "Breaking",
    3: "Ripe (first phase)",
    4: "Ripe (second phase)",
    5: "Overripe",
}

#: Display names used in tables and figures.
MODEL_LABELS = {
    "linear": "Linear regression",
    "tree": "Decision tree regression",
    "svr": "Support vector regression",
    "mlp": "Neural network regression",
    "knn": "kNN classifier",
}

#: Hyper-parameter grids searched by inner grouped cross-validation.
#: Linear regression has no hyper-parameters, which is one of its advantages.
PARAM_GRIDS = {
    "linear": {},
    "tree": {"decisiontreeregressor__max_depth": [2, 3, 4, 6, 8, None],
             "decisiontreeregressor__min_samples_leaf": [1, 5, 10, 20]},
    "svr": {"svr__C": [0.3, 1, 3, 10], "svr__epsilon": [0.1, 0.3],
            "svr__gamma": ["scale", 0.1]},
    "mlp": {"mlpregressor__hidden_layer_sizes": [(8,), (16,), (16, 8)],
            "mlpregressor__alpha": [1e-3, 1e-2, 1e-1]},
    "knn": {"kneighborsclassifier__n_neighbors": [3, 5, 9, 15, 21]},
}


def make_model(name: str):
    """Return an unfitted scikit-learn pipeline (standardise -> model).

    Parameters
    ----------
    name : {"linear", "tree", "svr", "mlp", "knn"}

    Raises
    ------
    ValueError
        For an unknown model name.
    """
    if name == "linear":
        est = LinearRegression()
    elif name == "tree":
        est = DecisionTreeRegressor(random_state=0)
    elif name == "svr":
        est = SVR(kernel="rbf")
    elif name == "mlp":
        est = MLPRegressor(max_iter=3000, random_state=0)
    elif name == "knn":
        est = KNeighborsClassifier(weights="distance")
    else:
        raise ValueError(f"Unknown model '{name}'")
    # Standardising (zero mean, unit variance) stops features measured in
    # degrees (hue) from dominating features measured as fractions 0-1.
    return make_pipeline(StandardScaler(), est)


def is_regressor(name: str) -> bool:
    """True for models that predict a continuous ripening index."""
    return name != "knn"


def fit_model(name: str, X, y, groups=None, n_splits: int = 5):
    """Fit a model, tuning its settings by grouped cross-validation if needed.

    Settings are chosen using the training data only (``groups`` = avocado IDs,
    so photos of one fruit never end up on both sides of a split).
    Regressors are tuned for lowest mean absolute error, the kNN classifier for
    highest macro-F1.

    Returns
    -------
    model : fitted estimator
    best_params : dict
        Chosen settings (empty for linear regression).
    """
    grid = PARAM_GRIDS[name]
    y_fit = np.asarray(y, float) if is_regressor(name) else np.asarray(y, int)
    if not grid or groups is None:
        return make_model(name).fit(X, y_fit), {}
    n_splits = min(n_splits, len(np.unique(groups)))
    search = GridSearchCV(make_model(name), grid, cv=GroupKFold(n_splits=n_splits),
                          scoring="neg_mean_absolute_error" if is_regressor(name) else "f1_macro")
    search.fit(X, y_fit, groups=groups)
    return search.best_estimator_, search.best_params_


def to_stage(index_pred) -> np.ndarray:
    """Convert continuous ripening-index predictions to integer stages 1-5.

    Values are rounded to the nearest integer and clipped to the valid range,
    so e.g. 0.7 -> 1, 3.4 -> 3, 3.6 -> 4 and 5.8 -> 5.
    """
    return np.clip(np.rint(np.asarray(index_pred, float)), 1, 5).astype(int)


def linear_coefficients(model, feature_names) -> dict:
    """Standardised coefficients of a fitted linear-regression pipeline.

    Because inputs are standardised, each coefficient is the change in
    predicted ripening index for a one-standard-deviation change in that
    feature, so their sizes can be compared directly.
    """
    lr = model[-1]
    return {"intercept": float(lr.intercept_),
            **{n: float(c) for n, c in zip(feature_names, lr.coef_)}}


def kiosk_message(index_pred: float, k_rate: float, t0: float) -> str:
    """Turn a predicted ripening index into the message a shopper would see.

    Parameters
    ----------
    index_pred : float
        Continuous predicted ripening index (any real number; it is rounded
        to a stage for the label).
    k_rate, t0 : float
        Fitted logistic parameters for room-temperature (20 degC) storage,
        used to estimate the number of days until the fruit is ready.
    """
    stage = int(to_stage(index_pred))
    name = STAGE_NAMES[stage]
    if stage in (3, 4):
        return f"{name}: ready to eat now (index {index_pred:.1f})"
    if stage == 5:
        return f"{name}: past best quality, not recommended (index {index_pred:.1f})"
    days = days_until_ready(index_pred, k_rate, t0)
    return f"{name}: about {days:.0f} more day(s) at room temperature (index {index_pred:.1f})"
