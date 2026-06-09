#!/usr/bin/env python3
"""
Baseline classifiers on metabolomics (mtb.tsv) data.

- Binary classifiers only (specify positive class per dataset)
- Leakage-safe stratified k-fold CV: VarianceThreshold + StandardScaler fit on
  train fold only; SMOTE applied in preprocessed space, also train-fold only
- Sampling strategies: No Sampling vs SMOTE
- Outputs: results CSV + heatmap comparing strategies across datasets

Usage
-----
python mtb_baseline_classifiers.py \
  --datasets \
    "KIM_ADENOMAS_2020:data/KIM/mtb.tsv:data/KIM/metadata.tsv:Adenoma,Carcinoma" \
    "KOSTIC_INFANTS_DIABETES_2015:data/KOSTIC/mtb.tsv:data/KOSTIC/metadata.tsv:case" \
    "SINHA_CRC_2016:data/SINHA/mtb.tsv:data/SINHA/metadata.tsv:1" \
  --sample-id Sample --label-col Study.Group \
  --out-dir mtb_classifier_results

Dataset spec: NAME:FEATURE_FILE:META_FILE:POSITIVE_CLASS(ES)
  POSITIVE_CLASS(ES): comma-separated class labels that map to 1 (disease/case).
  Everything else maps to 0.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    warnings.warn("imbalanced-learn not found — SMOTE will be skipped. "
                  "Install with: pip install imbalanced-learn")

RANDOM_STATE = 42
MODEL_ORDER = ["LogisticRegression", "LinearSVM", "RandomForest", "kNN", "GaussianNB"]

# ──────────────────────────── I/O ────────────────────────────

def load_dataset(feature_path: Path, meta_path: Path,
                 sample_id: str, label_col: str,
                 positive_classes: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    sep_f = "\t" if feature_path.suffix in (".tsv", ".txt") else ","
    sep_m = "\t" if meta_path.suffix in (".tsv", ".txt") else ","

    feat = pd.read_csv(feature_path, sep=sep_f, index_col=0)
    meta = pd.read_csv(meta_path, sep=sep_m)

    if sample_id in meta.columns:
        meta = meta.set_index(sample_id)

    common = feat.index.intersection(meta.index)
    if len(common) == 0:
        raise ValueError(f"No common sample IDs (feature idx sample: {list(feat.index[:3])}, "
                         f"meta idx sample: {list(meta.index[:3])})")

    feat = feat.loc[common].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    meta = meta.loc[common]

    if label_col not in meta.columns:
        raise ValueError(f"Label column '{label_col}' not found. Available: {list(meta.columns)}")

    pos_set = {p.strip().lower() for p in positive_classes}
    y_raw = meta[label_col].astype(str).str.strip().str.lower()
    y = pd.Series((y_raw.isin(pos_set)).astype(int), index=common, name="label")

    counts = y.value_counts().to_dict()
    pos_label = "+".join(sorted(positive_classes))
    print(f"  Samples: {len(common)}, Features: {feat.shape[1]}, "
          f"Positive ({pos_label}): {counts.get(1, 0)}, Negative: {counts.get(0, 0)}")
    return feat, y


# ──────────────────────────── models ────────────────────────────

def build_classifiers() -> dict:
    lr = LogisticRegression(max_iter=2000, class_weight="balanced",
                             solver="saga", random_state=RANDOM_STATE)
    svm_base = LinearSVC(C=0.5, tol=1e-3, max_iter=50000,
                          class_weight="balanced", random_state=RANDOM_STATE)
    svm = CalibratedClassifierCV(svm_base, method="sigmoid", cv=3)
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                 random_state=RANDOM_STATE, n_jobs=-1)
    knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
    gnb = GaussianNB()

    return {
        "LogisticRegression": lr,
        "LinearSVM": svm,
        "RandomForest": rf,
        "kNN": knn,
        "GaussianNB": gnb,
    }


# ──────────────────────────── CV ────────────────────────────

def run_cv(X: pd.DataFrame, y: pd.Series, n_splits: int,
           var_thresh: float, use_smote: bool, random_state: int) -> list[dict]:
    sampling_name = "SMOTE" if use_smote else "No Sampling"

    min_class_n = int(y.value_counts().min())
    effective_cv = min(n_splits, max(2, min_class_n))
    if effective_cv != n_splits:
        print(f"  [Warn] CV reduced to {effective_cv} (minority class n={min_class_n})")

    cv = StratifiedKFold(n_splits=effective_cv, shuffle=True, random_state=random_state)
    classifiers = build_classifiers()
    rows = []

    for model_name, clf_proto in classifiers.items():
        aucs, bals, f1s = [], [], []

        for train_idx, test_idx in cv.split(X.values, y.values):
            X_train_raw, X_test_raw = X.iloc[train_idx].values, X.iloc[test_idx].values
            y_train, y_test = y.values[train_idx], y.values[test_idx]

            # Preprocessing: fit on train only
            preproc = Pipeline([
                ("var", VarianceThreshold(threshold=var_thresh)),
                ("scale", StandardScaler()),
            ])
            X_train = preproc.fit_transform(X_train_raw)
            X_test  = preproc.transform(X_test_raw)

            # SMOTE: applied in preprocessed space, train only
            if use_smote and HAS_SMOTE:
                n_minority = int((y_train == 1).sum())
                k = max(1, min(5, n_minority - 1))
                smote = SMOTE(k_neighbors=k, random_state=random_state)
                try:
                    X_train, y_train = smote.fit_resample(X_train, y_train)
                except Exception as e:
                    print(f"    [Warn] SMOTE failed fold: {e}")

            clf = clone(clf_proto)
            try:
                clf.fit(X_train, y_train)
            except Exception as e:
                print(f"    [Error] {model_name} fit failed: {e}")
                continue

            try:
                proba = clf.predict_proba(X_test)[:, 1]
                aucs.append(roc_auc_score(y_test, proba))
            except Exception as e:
                print(f"    [Warn] AUC failed for {model_name}: {e}")

            try:
                y_pred = clf.predict(X_test)
                bals.append(balanced_accuracy_score(y_test, y_pred))
                f1s.append(f1_score(y_test, y_pred, zero_division=0))
            except Exception as e:
                print(f"    [Warn] Metrics failed for {model_name}: {e}")

        def ms(xs):
            xs = np.array(xs, dtype=float)
            return (float(np.nanmean(xs)), float(np.nanstd(xs))) if xs.size else (np.nan, np.nan)

        auc_m, auc_s = ms(aucs)
        bal_m, bal_s = ms(bals)
        f1_m,  f1_s  = ms(f1s)

        rows.append({
            "sampling": sampling_name,
            "model": model_name,
            "roc_auc_mean": auc_m, "roc_auc_std": auc_s,
            "bal_acc_mean": bal_m, "bal_acc_std": bal_s,
            "f1_mean": f1_m,       "f1_std": f1_s,
            "n_samples": len(X),
            "n_features": X.shape[1],
            "cv_folds": effective_cv,
        })
        print(f"    [{sampling_name:11s}] {model_name:20s}  "
              f"AUC {auc_m:.3f}±{auc_s:.3f}  "
              f"BalAcc {bal_m:.3f}±{bal_s:.3f}  "
              f"F1 {f1_m:.3f}±{f1_s:.3f}")

    return rows


# ──────────────────────────── plots ────────────────────────────

def _make_pivot(df: pd.DataFrame, metric_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (mean_pivot, std_pivot) with consistent model column order."""
    std_col = metric_col.replace("_mean", "_std")
    pivot = df.pivot(index="dataset", columns="model", values=metric_col)
    pivot_std = df.pivot(index="dataset", columns="model", values=std_col)
    cols = [m for m in MODEL_ORDER if m in pivot.columns]
    return pivot[cols], pivot_std[cols]


def plot_heatmap(all_results: pd.DataFrame, out_dir: Path):
    """
    Grid: rows = sampling strategies, cols = metrics.
    Each cell is a heatmap with datasets as rows, models as columns.
    """
    strategies = ["No Sampling", "SMOTE"]
    strategies = [s for s in strategies if s in all_results["sampling"].unique()]
    metrics = [
        ("ROC-AUC",          "roc_auc_mean"),
        ("Balanced Accuracy","bal_acc_mean"),
        ("F1",               "f1_mean"),
    ]

    n_strat = len(strategies)
    n_met   = len(metrics)
    n_ds    = all_results["dataset"].nunique()

    cell_h = max(2.5, n_ds * 0.55 + 1.2)
    cell_w = 4.5
    fig, axes = plt.subplots(n_strat, n_met,
                              figsize=(cell_w * n_met, cell_h * n_strat),
                              squeeze=False)

    for row_i, strat in enumerate(strategies):
        df_s = all_results[all_results["sampling"] == strat]
        for col_i, (metric_label, metric_col) in enumerate(metrics):
            ax = axes[row_i][col_i]
            pivot, pivot_std = _make_pivot(df_s, metric_col)

            annot = pivot.copy().astype(object)
            for r in pivot.index:
                for c in pivot.columns:
                    v = pivot.loc[r, c]
                    s = pivot_std.loc[r, c]
                    annot.loc[r, c] = f"{v:.3f}\n±{s:.3f}" if not np.isnan(v) else "—"

            sns.heatmap(
                pivot.astype(float), ax=ax,
                annot=annot, fmt="",
                cmap="YlOrRd", vmin=0.4, vmax=1.0,
                linewidths=0.5,
                cbar_kws={"label": metric_label, "shrink": 0.8},
                annot_kws={"size": 7.5},
            )
            ax.set_title(f"{strat} — {metric_label}", fontsize=10)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="x", rotation=35, labelsize=8)
            ax.tick_params(axis="y", rotation=0,  labelsize=8)

    fig.suptitle("Metabolomics Baseline Classifiers: No Sampling vs SMOTE (5-fold CV)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    out_path = out_dir / "classifier_performance_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nHeatmap saved → {out_path}")


def plot_smote_delta(all_results: pd.DataFrame, out_dir: Path):
    """Bar chart showing SMOTE improvement (Δ) over No Sampling per dataset/model."""
    if not all(s in all_results["sampling"].unique() for s in ["No Sampling", "SMOTE"]):
        return

    metrics = [("roc_auc_mean", "ΔAUC"), ("bal_acc_mean", "ΔBalAcc"), ("f1_mean", "ΔF1")]
    datasets = all_results["dataset"].unique()
    palette = sns.color_palette("Set2", len(datasets))

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), sharey=False)

    for ax, (metric_col, label) in zip(axes, metrics):
        base = all_results[all_results["sampling"] == "No Sampling"].set_index(["dataset", "model"])[metric_col]
        smote = all_results[all_results["sampling"] == "SMOTE"].set_index(["dataset", "model"])[metric_col]
        delta = (smote - base).reset_index()
        delta.columns = ["dataset", "model", "delta"]

        models = [m for m in MODEL_ORDER if m in delta["model"].unique()]
        x = np.arange(len(models))
        width = 0.8 / len(datasets)

        for i, (ds, col) in enumerate(zip(datasets, palette)):
            vals = [delta.loc[(delta["dataset"] == ds) & (delta["model"] == m), "delta"].values
                    for m in models]
            vals = [v[0] if len(v) else np.nan for v in vals]
            offset = (i - len(datasets) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width=width * 0.9, color=col, alpha=0.85, label=ds)

        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(f"SMOTE improvement: {label}", fontsize=10)
        if ax == axes[0]:
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("SMOTE Δ over No Sampling (positive = improvement)", fontsize=12)
    fig.tight_layout()
    out_path = out_dir / "smote_delta.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"SMOTE delta chart saved → {out_path}")


# ──────────────────────────── main ────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Binary baseline classifiers on metabolomics data")
    ap.add_argument(
        "--datasets", nargs="+", required=True,
        metavar="NAME:FEATURE_TSV:META_TSV:POSITIVE",
        help=("One entry per dataset: name:feature_file:meta_file:positive_class(es). "
              "POSITIVE is comma-separated class labels that map to 1 (e.g. 'Adenoma,Carcinoma')."),
    )
    ap.add_argument("--sample-id", type=str, default="Sample")
    ap.add_argument("--label-col", type=str, default="Study.Group")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--variance-threshold", type=float, default=0.0)
    ap.add_argument("--random-state", type=int, default=RANDOM_STATE)
    ap.add_argument("--out-dir", type=Path, default=Path("mtb_classifier_results"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for entry in args.datasets:
        parts = entry.split(":")
        if len(parts) != 4:
            raise SystemExit(
                f"Each --datasets entry must be NAME:FEATURE:META:POSITIVE, got: {entry}"
            )
        ds_name = parts[0]
        feat_path = Path(parts[1])
        meta_path = Path(parts[2])
        positive_classes = [p.strip() for p in parts[3].split(",")]

        print(f"\n{'─'*60}\n[{ds_name}]  positive={positive_classes}")

        try:
            X, y = load_dataset(feat_path, meta_path, args.sample_id,
                                 args.label_col, positive_classes)
        except Exception as e:
            print(f"  [Error] Skipping {ds_name}: {e}")
            continue

        for use_smote in [False, True]:
            if use_smote and not HAS_SMOTE:
                continue
            rows = run_cv(X, y, n_splits=args.cv,
                          var_thresh=args.variance_threshold,
                          use_smote=use_smote,
                          random_state=args.random_state)
            for r in rows:
                r["dataset"] = ds_name
            all_rows.extend(rows)

    if not all_rows:
        print("No results to save.")
        return

    results_df = pd.DataFrame(all_rows)
    csv_path = args.out_dir / "mtb_baseline_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults CSV → {csv_path}")

    plot_heatmap(results_df, args.out_dir)
    plot_smote_delta(results_df, args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
