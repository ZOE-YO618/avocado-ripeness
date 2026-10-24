"""
End-to-end experiment for the avocado ripeness project.

Usage (from the repository root)::

    python scripts/run_pipeline.py --image-dir "PATH/TO/Avocado Ripening Dataset" \
        --metadata "data/Avocado Ripening Dataset.xlsx"

Optional: ``--subset data/subset_files.csv`` restricts the run to the image
subset used in the report (otherwise every image found in ``--image-dir``
that is listed in the metadata is used).

Outputs (written to ``results/``):
    features_200.csv          colour features of every image
    metrics.json              all numbers quoted in the report
    figures/*.png             figures used in the report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from avocado_ripeness.baseline import ThresholdClassifier, hue_ripeness_score  # noqa: E402
from avocado_ripeness.classifier import (MODEL_LABELS, STAGE_NAMES, fit_model,  # noqa: E402
                                         is_regressor, kiosk_message, linear_coefficients, to_stage)
from avocado_ripeness.features import (FEATURE_NAMES, LIGHTING_INVARIANT_FEATURES,  # noqa: E402
                                       adjust_brightness, extract_features, load_image)

from avocado_ripeness.ripening_model import (days_until_ready, fit_logistic, q10,  # noqa: E402
                                             time_to_index)

INV_COLS = [FEATURE_NAMES.index(n) for n in LIGHTING_INVARIANT_FEATURES]
CLASSES = [1, 2, 3, 4, 5]
MODELS = ["linear", "tree", "svr", "mlp", "knn"]   # compared on lighting-invariant features
FINAL = "linear"                                    # selected model (see report section 4)
GROUP_LABEL = {"T10": "10 °C", "T20": "20 °C", "Tam": "Ambient"}

# Reference palette (validated categorical slots 1-3) and neutral inks
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAGENTA, C_GREEN = \
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
MODEL_COLOR = {"baseline": C_ORANGE, "linear": C_BLUE, "tree": C_AQUA, "svr": C_YELLOW,
               "mlp": C_MAGENTA, "knn": C_GREEN}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GROUP_COLOR = {"T10": C_BLUE, "T20": C_ORANGE, "Tam": C_AQUA}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titleweight": "bold", "axes.titlesize": 11, "lines.linewidth": 2,
})


# --------------------------------------------------------------------------- data
def load_metadata(path: Path) -> pd.DataFrame:
    """Read the dataset spreadsheet and add a 'side' column (a/b)."""
    df = pd.read_excel(path)
    df["side"] = df["File Name"].str.split("_").str[3]
    return df


def select_images(meta: pd.DataFrame, image_dir: Path, subset: Path | None) -> pd.DataFrame:
    """Keep metadata rows whose image file exists (and is in the subset, if given)."""
    rows = meta.copy()
    if subset is not None:
        keep = set(pd.read_csv(subset)["File Name"])
        rows = rows[rows["File Name"].isin(keep)]
    exists = rows["File Name"].apply(lambda f: (image_dir / f"{f}.jpg").exists())
    rows = rows[exists].reset_index(drop=True)
    if rows.empty:
        raise SystemExit(f"No images found in {image_dir}")
    return rows


def extract_all(rows: pd.DataFrame, image_dir: Path, size: int | None,
                brightness: float = 1.0) -> tuple[np.ndarray, float]:
    """Extract features for every row; return (X, mean seconds per image)."""
    X = np.empty((len(rows), len(FEATURE_NAMES)))
    t0 = time.perf_counter()
    for i, f in enumerate(rows["File Name"]):
        img = load_image(image_dir / f"{f}.jpg", size=size)
        if brightness != 1.0:
            img = adjust_brightness(img, brightness)
        X[i] = extract_features(img)
    return X, (time.perf_counter() - t0) / len(rows)


# --------------------------------------------------------------------------- metrics
def summarise(y_true, stage_pred, index_pred=None) -> dict:
    """Stage-level and (if available) continuous-index metrics.

    Stage metrics: accuracy, macro precision/recall/F1, within-one-stage
    accuracy, stage MAE and "ready to eat" (stage 3-4 vs not) accuracy.
    Index metrics (regressors only): MAE, RMSE and R^2 of the continuous
    prediction against the true index.
    """
    y_true = np.asarray(y_true)
    stage_pred = np.asarray(stage_pred)
    rep = classification_report(y_true, stage_pred, labels=CLASSES, output_dict=True, zero_division=0)
    out = {
        "accuracy": accuracy_score(y_true, stage_pred),
        "macro_precision": rep["macro avg"]["precision"],
        "macro_recall": rep["macro avg"]["recall"],
        "macro_f1": f1_score(y_true, stage_pred, labels=CLASSES, average="macro", zero_division=0),
        "within_one_accuracy": float(np.mean(np.abs(y_true - stage_pred) <= 1)),
        "stage_mae": mean_absolute_error(y_true, stage_pred),
        "ready_to_eat_accuracy": float(np.mean(np.isin(stage_pred, [3, 4]) == np.isin(y_true, [3, 4]))),
        "per_class": {str(c): {k: rep[str(c)][k] for k in ("precision", "recall", "f1-score", "support")}
                      for c in CLASSES},
    }
    if index_pred is not None:
        out["index_mae"] = mean_absolute_error(y_true, index_pred)
        out["index_rmse"] = float(np.sqrt(mean_squared_error(y_true, index_pred)))
        out["index_r2"] = r2_score(y_true, index_pred)
    return out


def baseline_cv(X, y, groups, Xtest=None, n_splits=5):
    """Out-of-fold predictions of the colour-threshold baseline."""
    Xtest = X if Xtest is None else Xtest
    hue_idx = FEATURE_NAMES.index("hue_mean_deg")
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        base = ThresholdClassifier().fit(hue_ripeness_score(X[tr, hue_idx]), y[tr])
        pred[te] = base.predict(hue_ripeness_score(Xtest[te, hue_idx]))
    return pred


def model_cv(name, X, y, groups, cols, Xtest=None, n_splits=5):
    """Grouped outer CV for one model; settings tuned inside each training fold.

    Returns out-of-fold predictions (continuous index for regressors, stage
    for kNN), the chosen settings per fold, and timing information.
    """
    Xtest = X if Xtest is None else Xtest
    pred = np.zeros(len(y))
    params, fit_s, pred_s = [], 0.0, 0.0
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        t0 = time.perf_counter()
        model, bp = fit_model(name, X[tr][:, cols], y[tr], groups[tr])
        fit_s += time.perf_counter() - t0
        t0 = time.perf_counter()
        pred[te] = model.predict(Xtest[te][:, cols])
        pred_s += time.perf_counter() - t0
        params.append({k.split("__")[-1]: (list(v) if isinstance(v, tuple) else v) for k, v in bp.items()})
    return pred, params, {"tuning_and_fit_s": fit_s, "predict_us_per_sample": pred_s / len(y) * 1e6}


def evaluate(name, y, pred):
    """Wrap ``summarise`` for a model's out-of-fold predictions."""
    if is_regressor(name):
        return summarise(y, to_stage(pred), pred)
    return summarise(y, pred.astype(int), pred)


def n_stored_values(model) -> int:
    """Rough model size: number of numbers the fitted model must store."""
    est = model[-1]
    if hasattr(est, "coef_") and not hasattr(est, "support_vectors_"):
        return int(np.size(est.coef_) + 1)
    if hasattr(est, "support_vectors_"):
        return int(est.support_vectors_.size + est.dual_coef_.size + 1)
    if hasattr(est, "tree_"):
        return int(est.tree_.node_count * 3)
    if hasattr(est, "coefs_"):
        return int(sum(w.size for w in est.coefs_) + sum(b.size for b in est.intercepts_))
    if hasattr(est, "_fit_X"):
        return int(est._fit_X.size)
    return -1


# --------------------------------------------------------------------------- figures
def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_confusion(cms, titles, path):
    fig, axes = plt.subplots(1, len(cms), figsize=(4.4 * len(cms), 4))
    for ax, cm, title in zip(axes, cms, titles):
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.grid(False)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if norm[i, j] > 0.55 else INK, fontsize=9)
        ax.set_xticks(range(5), CLASSES)
        ax.set_yticks(range(5), CLASSES)
        ax.set_xlabel("Predicted stage")
        ax.set_ylabel("True stage")
        ax.set_title(title)
    _save(fig, path)


def fig_curves(meta, fits, path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    t_line = np.linspace(1, 26, 300)
    for g in ("T10", "T20", "Tam"):
        sub = meta[meta["Storage Group"] == g]
        daily = sub.groupby("Day of Experiment")["Ripening Index Classification"].mean()
        ax.scatter(daily.index, daily.values, s=22, color=GROUP_COLOR[g],
                   edgecolor=SURFACE, linewidth=1, zorder=3)
        tmax = sub["Day of Experiment"].max()
        tl = t_line[t_line <= tmax]
        ax.plot(tl, fits[g]["bounded"].predict(tl), color=GROUP_COLOR[g], label=GROUP_LABEL[g])
    ax.axhspan(3, 4, color=GRID, alpha=0.6, zorder=0)
    ax.text(25.5, 3.5, "ready to eat", ha="right", va="center", color=INK2, fontsize=9)
    ax.set_xlabel("Day of experiment")
    ax.set_ylabel("Ripening index (1-5)")
    ax.set_ylim(0.8, 5.2)
    ax.set_title("Logistic ripening curves (lines) vs daily mean index (dots)")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, path)


def fig_bars(labels, series, ylabel, title, path, figsize=(7, 3.6), ncol=None, fmt="{:.2f}"):
    """Grouped bar chart. ``series`` is a list of (name, values, colour)."""
    x = np.arange(len(labels))
    n = len(series)
    w = 0.84 / n
    fig, ax = plt.subplots(figsize=figsize)
    for i, (name, vals, col) in enumerate(series):
        offs = (i - (n - 1) / 2) * w
        bars = ax.bar(x + offs, vals, w - 0.015, color=col, label=name)
        if n <= 3:
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, fmt.format(b.get_height()),
                        ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, ncol=ncol or n, loc="upper left", bbox_to_anchor=(0, 1.0), fontsize=8.5)
    ax.set_ylim(0, 1.18)
    _save(fig, path)


def fig_pred_vs_true(y, pred, path):
    """Strip plot of continuous linear-regression predictions per true stage."""
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for c in CLASSES:
        m = y == c
        ax.scatter(c + rng.uniform(-0.18, 0.18, m.sum()), pred[m], s=9, color=C_BLUE, alpha=0.45, linewidth=0)
        ax.plot([c - 0.28, c + 0.28], [np.median(pred[m])] * 2, color=INK, linewidth=2)
    ax.plot([0.6, 5.4], [0.6, 5.4], color=INK2, linestyle="--", linewidth=1, label="perfect prediction")
    for b in (1.5, 2.5, 3.5, 4.5):
        ax.axhline(b, color=GRID, linewidth=1, zorder=0)
    ax.set_xticks(CLASSES)
    ax.set_xlabel("True ripening stage")
    ax.set_ylabel("Predicted ripening index")
    ax.set_title("Linear regression: predictions on unseen avocados")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    _save(fig, path)


def fig_coefficients(coefs, path):
    names = [n for n in LIGHTING_INVARIANT_FEATURES]
    vals = np.array([coefs[n] for n in names])
    order = np.argsort(np.abs(vals))
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.barh(np.array(names)[order], vals[order], color=[C_BLUE if v > 0 else C_ORANGE for v in vals[order]])
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_xlabel("Change in predicted index per 1 standard deviation")
    ax.set_title("Linear regression coefficients (standardised)")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def fig_resolution(res, path):
    sizes = [r["size_label"] for r in res]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(sizes, [r["index_mae"] for r in res], marker="o", color=C_BLUE)
    axes[0].set_ylabel("Linear regression MAE (stages)")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Error vs input resolution")
    axes[1].plot(sizes, [r["ms_per_image"] for r in res], marker="o", color=C_BLUE)
    axes[1].set_ylabel("Feature extraction time (ms / image)")
    axes[1].set_title("Runtime vs input resolution")
    for ax in axes:
        ax.set_xlabel("Image size (pixels per side)")
    _save(fig, path)


def fig_examples(rows, image_dir, path):
    from avocado_ripeness.features import fruit_mask
    picks = [rows[rows["Ripening Index Classification"] == c].iloc[0]["File Name"] for c in CLASSES]
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.4))
    for j, f in enumerate(picks):
        img = load_image(image_dir / f"{f}.jpg", size=200)
        m = fruit_mask(img)
        shown = img.copy()
        shown[~m] = (shown[~m] * 0.15).astype(np.uint8)
        axes[0, j].imshow(img)
        axes[1, j].imshow(shown)
        axes[0, j].set_title(f"Stage {CLASSES[j]}\n{STAGE_NAMES[CLASSES[j]]}", fontsize=9)
        for a in axes[:, j]:
            a.axis("off")
    axes[0, 0].text(-0.1, 0.5, "Photo", transform=axes[0, 0].transAxes, rotation=90, va="center", ha="right")
    axes[1, 0].text(-0.1, 0.5, "Fruit mask", transform=axes[1, 0].transAxes, rotation=90, va="center", ha="right")
    _save(fig, path)


# --------------------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image-dir", required=True, type=Path)
    p.add_argument("--metadata", default=ROOT / "data" / "Avocado Ripening Dataset.xlsx", type=Path)
    p.add_argument("--subset", default=None, type=Path)
    p.add_argument("--out", default=ROOT / "results", type=Path)
    p.add_argument("--skip-resolution", action="store_true", help="skip the slow resolution study")
    args = p.parse_args(argv)

    out, figs = args.out, args.out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    metrics: dict = {}

    # 1. Data --------------------------------------------------------------
    meta = load_metadata(args.metadata)
    rows = select_images(meta, args.image_dir, args.subset)
    y = rows["Ripening Index Classification"].to_numpy()
    groups = rows["Sample"].to_numpy()
    metrics["data"] = {
        "n_images": int(len(rows)), "n_avocados": int(rows["Sample"].nunique()),
        "class_counts": {str(c): int((y == c).sum()) for c in CLASSES},
        "storage_counts": rows["Storage Group"].value_counts().to_dict(),
    }
    print(f"Using {len(rows)} images of {rows['Sample'].nunique()} avocados")

    # 2. Features ----------------------------------------------------------
    X, sec = extract_all(rows, args.image_dir, size=200)
    pd.concat([rows, pd.DataFrame(X, columns=FEATURE_NAMES)], axis=1).to_csv(out / "features_200.csv", index=False)
    metrics["feature_extraction_ms_per_image_200px"] = sec * 1000
    fig_examples(rows, args.image_dir, figs / "examples_segmentation.png")

    # 3. Model comparison, grouped 5-fold CV (test avocados never seen in training)
    pred_base = baseline_cv(X, y, groups)
    metrics["baseline"] = summarise(y, pred_base)
    preds, comparison = {}, {}
    for name in MODELS:
        print(f"  cross-validating {MODEL_LABELS[name]} ...")
        preds[name], params, timing = model_cv(name, X, y, groups, INV_COLS)
        comparison[name] = {**evaluate(name, y, preds[name]), "chosen_params_per_fold": params, **timing}
    metrics["models"] = comparison
    final_stage = to_stage(preds[FINAL])

    # Feature-set comparison for the final model (all 10 features vs lighting-invariant 7)
    pred_lin_all, _, _ = model_cv(FINAL, X, y, groups, list(range(X.shape[1])))
    metrics["linear_all_features"] = evaluate(FINAL, y, pred_lin_all)

    cms = [confusion_matrix(y, p_, labels=CLASSES) for p_ in (pred_base, final_stage)]
    metrics["confusion_baseline"], metrics["confusion_final"] = [c.tolist() for c in cms]
    fig_confusion(cms, ["Baseline: colour threshold", "Final: linear regression"], figs / "confusion_matrices.png")
    fig_pred_vs_true(y, preds[FINAL], figs / "linear_pred_vs_true.png")

    def _row(m):
        return [m["accuracy"], m["within_one_accuracy"], m["ready_to_eat_accuracy"], m.get("index_r2", np.nan)]
    series = [("Baseline", _row(metrics["baseline"]), MODEL_COLOR["baseline"])]
    series += [(MODEL_LABELS[n].replace(" regression", ""), _row(comparison[n]), MODEL_COLOR[n]) for n in MODELS]
    fig_bars(["Stage accuracy", "Within ±1 stage", "Ready-to-eat\n(yes/no)", "R² of index"], series,
             "Score (grouped 5-fold CV)", "Model comparison on unseen avocados", figs / "model_comparison.png",
             figsize=(8, 3.8), ncol=3)

    # 4. Final model: coefficients, size, per-scan latency --------------------
    final_model, _ = fit_model(FINAL, X[:, INV_COLS], y)
    metrics["final_model"] = {"name": FINAL, "features": LIGHTING_INVARIANT_FEATURES,
                              "coefficients": linear_coefficients(final_model, LIGHTING_INVARIANT_FEATURES)}
    fig_coefficients(metrics["final_model"]["coefficients"], figs / "linear_coefficients.png")
    metrics["model_size_values"] = {n: n_stored_values(fit_model(n, X[:, INV_COLS], y, groups)[0]) for n in MODELS}
    sample_files = rows["File Name"].iloc[:50]
    t0 = time.perf_counter()
    for f in sample_files:
        feats = extract_features(load_image(args.image_dir / f"{f}.jpg", 200))
        final_model.predict(feats[None, INV_COLS])
    metrics["per_scan_latency_ms"] = (time.perf_counter() - t0) / len(sample_files) * 1000

    # 5. Robustness to lighting --------------------------------------------
    factors = [0.7, 0.85, 1.0, 1.15, 1.3]
    robust = {"factor": factors, "baseline_acc": [], "linear_all_acc": [], "linear_invariant_acc": [],
              "linear_invariant_mae": []}
    for fct in factors:
        Xp = X if fct == 1.0 else extract_all(rows, args.image_dir, 200, brightness=fct)[0]
        robust["baseline_acc"].append(accuracy_score(y, baseline_cv(X, y, groups, Xtest=Xp)))
        pa, _, _ = model_cv(FINAL, X, y, groups, list(range(X.shape[1])), Xtest=Xp)
        pi, _, _ = model_cv(FINAL, X, y, groups, INV_COLS, Xtest=Xp)
        robust["linear_all_acc"].append(accuracy_score(y, to_stage(pa)))
        robust["linear_invariant_acc"].append(accuracy_score(y, to_stage(pi)))
        robust["linear_invariant_mae"].append(mean_absolute_error(y, pi))
    metrics["robustness_brightness"] = robust
    fig_bars([f"×{f}" for f in factors],
             [("Baseline", robust["baseline_acc"], C_ORANGE),
              ("Linear regression, all 10 features", robust["linear_all_acc"], C_AQUA),
              ("Linear regression, lighting-invariant", robust["linear_invariant_acc"], C_BLUE)],
             "Stage accuracy", "Sensitivity to lighting: test images darkened/brightened",
             figs / "robustness_brightness.png", ncol=2)

    # 6. Resolution / runtime trade-off -------------------------------------
    if not args.skip_resolution:
        res = []
        for size in (25, 50, 100, 200, 400, 800):
            Xs, sec_s = extract_all(rows, args.image_dir, size=size if size != 800 else None)
            ps, _, _ = model_cv(FINAL, Xs, y, groups, INV_COLS)
            res.append({"size_label": str(size), "ms_per_image": sec_s * 1000,
                        "index_mae": mean_absolute_error(y, ps), "stage_accuracy": accuracy_score(y, to_stage(ps))})
            print(f"  size {size}: {sec_s*1000:.1f} ms/img, MAE {res[-1]['index_mae']:.3f}")
        metrics["resolution_study"] = res
        fig_resolution(res, figs / "resolution_tradeoff.png")

    # 7. Logistic ripening model on the FULL metadata (14,722 labels) ----------
    fits, curve_metrics = {}, {}
    for g in ("T10", "T20", "Tam"):
        sub = meta[meta["Storage Group"] == g]
        t, r = sub["Day of Experiment"].to_numpy(), sub["Ripening Index Classification"].to_numpy()
        fits[g] = {"3p": fit_logistic(t, r, "3p"), "bounded": fit_logistic(t, r, "bounded")}
        kb, t0b = fits[g]["bounded"].params["k"], fits[g]["bounded"].params["t0"]
        curve_metrics[g] = {
            m: {"params": fits[g][m].params, "std_errors": fits[g][m].std_errors,
                "rmse": fits[g][m].rmse, "r2": fits[g][m].r2, "n": fits[g][m].n} for m in ("3p", "bounded")
        }
        curve_metrics[g]["day_reaches_stage3"] = float(time_to_index(3.0, kb, t0b))
        curve_metrics[g]["day_reaches_stage4_5"] = float(time_to_index(4.5, kb, t0b))

        # Validation: fit on 80 % of avocados, test on the held-out 20 %
        rng = np.random.default_rng(0)
        ids = sub["Sample"].unique()
        test_ids = rng.choice(ids, size=max(1, len(ids) // 5), replace=False)
        trm = ~sub["Sample"].isin(test_ids)
        fh = fit_logistic(t[trm], r[trm], "bounded")
        resid = r[~trm] - fh.predict(t[~trm])
        curve_metrics[g]["holdout_rmse"] = float(np.sqrt(np.mean(resid ** 2)))
    curve_metrics["q10_T10_to_T20"] = q10(curve_metrics["T10"]["bounded"]["params"]["k"],
                                          curve_metrics["T20"]["bounded"]["params"]["k"])
    metrics["ripening_curves"] = curve_metrics
    fig_curves(meta, fits, figs / "ripening_curves.png")

    # 8. Kiosk demo -----------------------------------------------------------
    kr, t0r = fits["T20"]["bounded"].params["k"], fits["T20"]["bounded"].params["t0"]
    demo = []
    for c in CLASSES:
        i = int(np.where(y == c)[0][0])
        demo.append({"file": rows["File Name"].iloc[i], "true_stage": int(c),
                     "predicted_index": float(preds[FINAL][i]),
                     "message": kiosk_message(float(preds[FINAL][i]), kr, t0r)})
    metrics["kiosk_demo"] = demo
    metrics["days_until_ready_at_20C"] = {str(s): days_until_ready(s, kr, t0r) for s in (1.5, 2.0, 2.5)}

    with open(out / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)

    print("\nModel comparison (grouped 5-fold CV, lighting-invariant features):")
    b = metrics["baseline"]
    print(f"  {'Baseline':28s} stage acc {b['accuracy']:.3f}  within±1 {b['within_one_accuracy']:.3f}  "
          f"ready {b['ready_to_eat_accuracy']:.3f}")
    for n in MODELS:
        m = comparison[n]
        print(f"  {MODEL_LABELS[n]:28s} stage acc {m['accuracy']:.3f}  within±1 {m['within_one_accuracy']:.3f}  "
              f"ready {m['ready_to_eat_accuracy']:.3f}  MAE {m['index_mae']:.3f}  R2 {m['index_r2']:.3f}")
    print("Kiosk demo:")
    for d in demo:
        print("  ", d)
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
