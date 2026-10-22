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
                             f1_score, mean_absolute_error)
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from avocado_ripeness.baseline import ThresholdClassifier, hue_ripeness_score  # noqa: E402
from avocado_ripeness.classifier import STAGE_NAMES, kiosk_message, make_knn, select_k  # noqa: E402
from avocado_ripeness.features import (FEATURE_NAMES, LIGHTING_INVARIANT_FEATURES,  # noqa: E402
                                       adjust_brightness, extract_features, load_image)

INV_COLS = [FEATURE_NAMES.index(n) for n in LIGHTING_INVARIANT_FEATURES]
from avocado_ripeness.ripening_model import (days_until_ready, fit_logistic, q10,  # noqa: E402
                                             time_to_index)

CLASSES = [1, 2, 3, 4, 5]
GROUP_LABEL = {"T10": "10 °C", "T20": "20 °C", "Tam": "Ambient"}

# Reference palette (validated categorical slots 1-3) and neutral inks
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
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
def summarise(y_true, y_pred) -> dict:
    """Accuracy, macro precision/recall/F1, within-one-stage accuracy, stage MAE."""
    rep = classification_report(y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": rep["macro avg"]["precision"],
        "macro_recall": rep["macro avg"]["recall"],
        "macro_f1": f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0),
        "within_one_accuracy": float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= 1)),
        "stage_mae": mean_absolute_error(y_true, y_pred),
        "per_class": {str(c): {k: rep[str(c)][k] for k in ("precision", "recall", "f1-score", "support")}
                      for c in CLASSES},
    }


def run_cv(X, y, groups, n_splits=5, cols=None):
    """Outer grouped CV: baseline vs kNN (k chosen by inner grouped CV).

    ``cols`` selects which feature columns the kNN uses (default: all).
    """
    cols = list(range(X.shape[1])) if cols is None else cols
    outer = GroupKFold(n_splits=n_splits)
    pred_base = np.zeros_like(y)
    pred_knn = np.zeros_like(y)
    proba_knn = np.zeros(len(y))
    chosen_k, k_curves = [], []
    hue_idx = FEATURE_NAMES.index("hue_mean_deg")
    for tr, te in outer.split(X, y, groups):
        base = ThresholdClassifier().fit(hue_ripeness_score(X[tr, hue_idx]), y[tr])
        pred_base[te] = base.predict(hue_ripeness_score(X[te, hue_idx]))

        k, curve = select_k(X[tr][:, cols], y[tr], groups[tr])
        chosen_k.append(k)
        k_curves.append(curve)
        model = make_knn(k).fit(X[tr][:, cols], y[tr])
        pred_knn[te] = model.predict(X[te][:, cols])
        proba_knn[te] = model.predict_proba(X[te][:, cols]).max(axis=1)
    return pred_base, pred_knn, proba_knn, chosen_k, k_curves


# --------------------------------------------------------------------------- figures
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
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_k_selection(k_curves, path):
    ks = sorted(k_curves[0])
    vals = np.array([[c[k] for k in ks] for c in k_curves])
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(ks, vals.mean(0), color=C_BLUE, marker="o", markersize=5)
    ax.fill_between(ks, vals.min(0), vals.max(0), color=C_BLUE, alpha=0.15, linewidth=0)
    ax.set_xlabel("Number of neighbours k")
    ax.set_ylabel("Inner-CV macro F1")
    ax.set_title("Choosing k (mean and range over outer folds)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


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
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_bars(labels, series, ylabel, title, path):
    """Grouped bar chart. ``series`` is a list of (name, values, colour)."""
    x = np.arange(len(labels))
    n = len(series)
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(7, 3.6))
    all_bars = []
    for i, (name, vals, col) in enumerate(series):
        offs = (i - (n - 1) / 2) * w
        all_bars.append(ax.bar(x + offs, vals, w - 0.02, color=col, label=name))
    for bars in all_bars:
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.2f}",
                    ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, ncol=n, loc="upper left", bbox_to_anchor=(0, 1.0))
    ax.set_ylim(0, 1.15)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_resolution(res, path):
    sizes = [r["size_label"] for r in res]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(sizes, [r["knn_macro_f1"] for r in res], marker="o", color=C_BLUE)
    axes[0].set_ylabel("kNN (lighting-invariant) macro F1")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Accuracy vs input resolution")
    axes[1].plot(sizes, [r["ms_per_image"] for r in res], marker="o", color=C_BLUE)
    axes[1].set_ylabel("Feature extraction time (ms / image)")
    axes[1].set_title("Runtime vs input resolution")
    for ax in axes:
        ax.set_xlabel("Image size (pixels per side)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


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
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


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

    # 3. Baseline vs kNN (all features) vs kNN (lighting-invariant), grouped 5-fold CV
    pred_base, pred_knn, proba_knn, chosen_k, k_curves = run_cv(X, y, groups)
    _, pred_inv, proba_inv, chosen_k_inv, _ = run_cv(X, y, groups, cols=INV_COLS)
    metrics["baseline"] = summarise(y, pred_base)
    metrics["knn"] = summarise(y, pred_knn)
    metrics["knn"]["chosen_k_per_fold"] = chosen_k
    metrics["knn_invariant"] = summarise(y, pred_inv)
    metrics["knn_invariant"]["chosen_k_per_fold"] = chosen_k_inv
    cms = [confusion_matrix(y, p_, labels=CLASSES) for p_ in (pred_base, pred_knn, pred_inv)]
    metrics["confusion_baseline"], metrics["confusion_knn"], metrics["confusion_knn_invariant"] = \
        [c.tolist() for c in cms]
    fig_confusion(cms, ["Baseline: colour threshold", "kNN (all 10 features)", "kNN (lighting-invariant)"],
                  figs / "confusion_matrices.png")
    fig_k_selection(k_curves, figs / "k_selection.png")

    # Consumer-level view: is the fruit "ready to eat" (stage 3-4) or not?
    ready_true = np.isin(y, [3, 4])
    metrics["ready_to_eat_accuracy"] = {
        name: float(np.mean(np.isin(p_, [3, 4]) == ready_true))
        for name, p_ in (("baseline", pred_base), ("knn", pred_knn), ("knn_invariant", pred_inv))
    }
    def _row(key):
        m = metrics[key]
        return [m["accuracy"], m["macro_f1"], m["within_one_accuracy"], metrics["ready_to_eat_accuracy"][key]]
    fig_bars(["Accuracy", "Macro F1", "Within ±1 stage", "Ready-to-eat\n(yes/no)"],
             [("Baseline", _row("baseline"), C_ORANGE), ("kNN all features", _row("knn"), C_BLUE),
              ("kNN lighting-invariant", _row("knn_invariant"), C_AQUA)],
             "Score (grouped 5-fold CV)", "Classifier comparison on unseen avocados", figs / "baseline_vs_knn.png")

    # 4. Per-scan latency (feature extraction + prediction) ------------------
    k_final = int(pd.Series(chosen_k_inv).mode()[0])
    final_model = make_knn(k_final).fit(X[:, INV_COLS], y)
    sample_files = rows["File Name"].iloc[:50]
    t0 = time.perf_counter()
    for f in sample_files:
        feats = extract_features(load_image(args.image_dir / f"{f}.jpg", 200))
        final_model.predict(feats[None, INV_COLS])
    metrics["per_scan_latency_ms"] = (time.perf_counter() - t0) / len(sample_files) * 1000
    metrics["final_k"] = k_final

    # 5. Robustness to lighting --------------------------------------------
    factors = [0.7, 0.85, 1.0, 1.15, 1.3]
    robust = {"factor": factors, "baseline_acc": [], "knn_acc": [], "knn_invariant_acc": []}
    hue_idx = FEATURE_NAMES.index("hue_mean_deg")
    for fct in factors:
        Xp = X if fct == 1.0 else extract_all(rows, args.image_dir, 200, brightness=fct)[0]
        pb, pk, pi = np.zeros_like(y), np.zeros_like(y), np.zeros_like(y)
        for (tr, te), k, ki in zip(GroupKFold(5).split(X, y, groups), chosen_k, chosen_k_inv):
            base = ThresholdClassifier().fit(hue_ripeness_score(X[tr, hue_idx]), y[tr])
            pb[te] = base.predict(hue_ripeness_score(Xp[te, hue_idx]))
            pk[te] = make_knn(k).fit(X[tr], y[tr]).predict(Xp[te])
            pi[te] = make_knn(ki).fit(X[tr][:, INV_COLS], y[tr]).predict(Xp[te][:, INV_COLS])
        robust["baseline_acc"].append(accuracy_score(y, pb))
        robust["knn_acc"].append(accuracy_score(y, pk))
        robust["knn_invariant_acc"].append(accuracy_score(y, pi))
    metrics["robustness_brightness"] = robust
    fig_bars([f"×{f}" for f in factors],
             [("Baseline", robust["baseline_acc"], C_ORANGE), ("kNN all features", robust["knn_acc"], C_BLUE),
              ("kNN lighting-invariant", robust["knn_invariant_acc"], C_AQUA)], "Accuracy",
             "Sensitivity to lighting: test images brightened/darkened", figs / "robustness_brightness.png")

    # 6. Resolution / runtime trade-off -------------------------------------
    if not args.skip_resolution:
        res = []
        for size in (25, 50, 100, 200, 400, 800):
            Xs, sec_s = extract_all(rows, args.image_dir, size=size if size != 800 else None)
            _, pk, _, _, _ = run_cv(Xs, y, groups, cols=INV_COLS)
            res.append({"size_label": str(size), "ms_per_image": sec_s * 1000,
                        "knn_macro_f1": f1_score(y, pk, average="macro"),
                        "knn_accuracy": accuracy_score(y, pk)})
            print(f"  size {size}: {sec_s*1000:.1f} ms/img, F1 {res[-1]['knn_macro_f1']:.3f}")
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
        stage = int(pred_inv[i])
        demo.append({"file": rows["File Name"].iloc[i], "true_stage": int(c), "predicted_stage": stage,
                     "message": kiosk_message(stage, float(proba_inv[i]), kr, t0r)})
    metrics["kiosk_demo"] = demo
    metrics["days_until_ready_at_20C"] = {str(s): days_until_ready(s, kr, t0r) for s in (1, 2)}

    with open(out / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    for key in ("baseline", "knn", "knn_invariant"):
        m = metrics[key]
        print(f"{key:15s} acc {m['accuracy']:.3f}  macroF1 {m['macro_f1']:.3f}  "
              f"within1 {m['within_one_accuracy']:.3f}  MAE {m['stage_mae']:.3f}")
    print("Kiosk demo:")
    for d in demo:
        print("  ", d)
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
