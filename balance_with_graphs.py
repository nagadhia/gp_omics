#!/usr/bin/env python3
"""
  1) Read an OTU/species/genera feature table + metadata
  2) Infer binary labels
  3) Build a balanced dataset in original abundance space (pre-VarianceThreshold, no scaling)
     using either AdaLoRAS (default), SMOTE, or NoSampling
  4) Save balanced feature table + metadata with additional `is_synthetic` column
  5) (Optional) Train baseline classifiers with leakage-safe CV with oversampling before scaling
  6) (Optional) Run your per-sample graph-generation script in on the balanced dataset


Usage example:

python balance_and_graphs_abundance.py \
  --feature-file otu.csv \
  --meta-file meta.csv \
  --label-col Condition \
  --positive "case,disease" \
  --out-dir balanced_out \
  --sampler AdaLoRAS \
  --run-graphs \
  --graphs-script graphs.py \
  --graphs-out graphs_out \
  --graphs-sample-id Sample \
  --graphs-condition-id Study.Group

Depdent on graph builder script available as a CLI , called with `python <graphs-script> bootstrap ...`

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import subprocess
import sys

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, accuracy_score,
)
from sklearn.neighbors import NearestNeighbors

RANDOM_STATE = 42

#AdaLoRAS (abundance-space) 
class AdaLoRASOversampler:
    """Hybrid ADASYN + LoRAS oversampler (operates in input feature space).
    """

    def __init__(self, minority_class=1, beta=1.0, k_all=5, k_min=20,
                 n_shadows=50, affine_dim=5, noise_scale=0.15, convex=True,
                 feature_clip: str | Tuple[float, float] | None = "minmax",
                 random_state: Optional[int] = None):
        self.minority_class = minority_class
        self.beta = float(beta)
        self.k_all = int(k_all)
        self.k_min = int(k_min)
        self.n_shadows = int(n_shadows)
        self.affine_dim = int(affine_dim)
        self.noise_scale = float(noise_scale)
        self.convex = bool(convex)
        self.feature_clip = feature_clip
        self.random_state = random_state

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        X_min = X[y == self.minority_class]
        X_maj = X[y != self.minority_class]
        n_min, n_maj = len(X_min), len(X_maj)

        def _return(Xr, yr):
            return Xr.copy(), yr.copy()

        if n_min == 0 or n_maj == 0 or self.beta <= 0:
            return _return(X, y)
        gap = n_maj - n_min
        if gap <= 0:
            return _return(X, y)

        G = int(np.floor(gap * self.beta))
        if G == 0:
            return _return(X, y)

        k_all_eff = max(1, min(self.k_all, len(X) - 1))
        knn_all = NearestNeighbors(n_neighbors=k_all_eff).fit(X)

        k_min_eff = max(1, min(self.k_min, max(1, n_min - 1)))
        knn_min = NearestNeighbors(n_neighbors=k_min_eff).fit(X_min)

        # Robust per-feature shadow noise based on IQR (minority only)
        q25, q75 = np.percentile(X_min, [25, 75], axis=0)
        iqr = np.maximum(q75 - q25, 1e-12)
        shadow_sigma = self.noise_scale * iqr

        fmin = fmax = lo = hi = None
        if self.feature_clip == "minmax":
            fmin = X.min(axis=0)
            fmax = X.max(axis=0)
        elif isinstance(self.feature_clip, (tuple, list)) and len(self.feature_clip) == 2:
            lo, hi = self.feature_clip

        # Local difficulty r[i] from all-neighborhood composition
        r = np.zeros(n_min, dtype=float)
        for i, xi in enumerate(X_min):
            _, idxs = knn_all.kneighbors(xi.reshape(1, -1))
            neigh_labels = y[idxs[0]]
            r[i] = np.sum(neigh_labels != self.minority_class) / k_all_eff

        d = (r / r.sum()) if r.sum() > 0 else np.full(n_min, 1.0 / n_min)
        raw = d * G
        g = np.floor(raw).astype(int)
        rem = G - g.sum()
        if rem > 0:
            order = np.argsort(-(raw - g))
            g[order[:rem]] += 1

        # cache minority neighborhoods
        min_neighbor_cache: Dict[int, np.ndarray] = {}
        for i, xi in enumerate(X_min):
            _, idxs = knn_min.kneighbors(xi.reshape(1, -1))
            min_neighbor_cache[i] = X_min[idxs[0]]

        synth = []
        for i, gi in enumerate(g):
            if gi <= 0:
                continue
            neigh = min_neighbor_cache[i]
            if len(neigh) == 0:
                continue
            for _ in range(gi):
                chosen = neigh[rng.integers(0, len(neigh), size=self.n_shadows)]
                noise = rng.normal(0.0, 1.0, size=chosen.shape) * shadow_sigma
                shadows = chosen + noise
                m = min(self.affine_dim, len(shadows))
                idx = rng.choice(len(shadows), size=m, replace=False)
                S = shadows[idx]
                if self.convex:
                    w = rng.dirichlet(np.ones(m))
                else:
                    w = rng.normal(size=m)
                    if np.allclose(w, 0):
                        w[0] = 1.0
                    w = w / (w.sum() + 1e-12)
                x_syn = (S * w[:, None]).sum(axis=0)
                if self.feature_clip is not None:
                    if self.feature_clip == "minmax":
                        x_syn = np.clip(x_syn, fmin, fmax)
                    else:
                        x_syn = np.clip(x_syn, lo, hi)
                synth.append(x_syn)

        if not synth:
            return _return(X, y)

        X_synth = np.vstack(synth)
        y_synth = np.full(len(synth), self.minority_class, dtype=y.dtype)
        X_res = np.vstack([X, X_synth])
        y_res = np.concatenate([y, y_synth])
        return X_res, y_res

def _read_any_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, index_col=0, sep=sep)

def read_tables(feature_file: Path, meta_file: Path, id_col: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X = _read_any_table(feature_file)
    meta = _read_any_table(meta_file)

    X.index = X.index.astype(str).str.strip()

    if isinstance(meta.index, pd.MultiIndex):
        meta = meta.reset_index(drop=False)
    else:
        meta.index = meta.index.astype(str).str.strip()

    if id_col is not None:
        if id_col not in meta.columns:
            raise SystemExit(f"--id-col '{id_col}' not found in metadata columns: {list(meta.columns)}")
        meta[id_col] = meta[id_col].astype(str).str.strip()
        meta = meta.set_index(id_col, drop=False)

    # try common ID columns if no overlap
    common = X.index.intersection(meta.index.astype(str))
    if len(common) == 0:
        candidates = [c for c in ["Sample", "sample", "SampleID", "Subject", "ID"] if c in meta.columns]
        for c in candidates:
            vals = meta[c].astype(str).str.strip()
            overlap = len(set(X.index).intersection(set(vals)))
            if overlap > 0:
                meta[c] = vals
                meta = meta.set_index(c, drop=False)
                break
        common = X.index.intersection(meta.index.astype(str))

    if len(common) == 0:
        raise SystemExit("No overlapping sample IDs between features and metadata after alignment. Use --id-col.")

    X = X.loc[common].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    meta = meta.loc[common]
    return X, meta

def infer_labels(meta: pd.DataFrame, label_col: str, positive: str) -> pd.Series:
    if label_col not in meta.columns:
        raise SystemExit(f"--label-col '{label_col}' not found in metadata columns: {list(meta.columns)}")
    y_raw = meta[label_col].astype(str).str.strip()
    pos_values = {p.strip().lower() for p in positive.split(",")}
    y = y_raw.str.lower().isin(pos_values).astype(int)
    if y.nunique() != 2:
        raise SystemExit(f"Labels not binary after mapping, got {y.value_counts().to_dict()}")
    y.index = meta.index
    return y

def make_balanced_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    sampler_name: str,
    var_threshold: float,
    random_state: int,
    beta: float,
    k_all: int,
    k_min: int,
    n_shadows: int,
    affine_dim: int,
    noise_scale: float,
) -> Tuple[pd.DataFrame, pd.Series, List[str], np.ndarray]:
    
    """Return (X_balanced, y_balanced, kept_feature_names, is_synthetic_mask).

    Steps:
      - Fit VarianceThreshold on abundance space; transform X
      - Oversample in this abundance space (no scaling)
      - Preserve original sample IDs for real rows; create SYNxxxxx for synthetic
    """
    
    # VarianceThreshold to drop constants in the abundance space
    vt = VarianceThreshold(threshold=var_threshold)
    Xv = vt.fit_transform(X.values)
    keep_mask = vt.get_support()
    kept_names = [c for c, keep in zip(list(X.columns), keep_mask) if keep]

    # Build sampler
    minority_class = int(y.value_counts().idxmin())
    if sampler_name.lower() == "adaloras":
        sampler = AdaLoRASOversampler(
            minority_class=minority_class,
            beta=beta,
            k_all=k_all,
            k_min=k_min,
            n_shadows=n_shadows,
            affine_dim=affine_dim,
            noise_scale=noise_scale,
            feature_clip="minmax",
            random_state=random_state,
        )
        X_res, y_res = sampler.fit_resample(Xv, y.values)
    elif sampler_name.lower() == "smote":
        n_min = int((y == minority_class).sum())
        if n_min <= 1:
            print("[WARN] Minority class has <=1 sample; skipping SMOTE (returning original X).")
            X_res, y_res = Xv, y.values
        else:
            k_smote = max(1, min(5, n_min - 1))
            smote = SMOTE(k_neighbors=k_smote, random_state=random_state)
            X_res, y_res = smote.fit_resample(Xv, y.values)
    elif sampler_name.lower() in {"none", "nosampling", "no sampling"}:
        X_res, y_res = Xv, y.values
    else:
        raise SystemExit(f"Unknown sampler: {sampler_name}")

    # Build DataFrame with kept feature names
    X_bal = pd.DataFrame(X_res, columns=kept_names)

    # is_synthetic mask and Sample IDs
    orig_n = len(X)
    is_syn = np.zeros(len(y_res), dtype=int)
    if len(y_res) > orig_n:
        is_syn[orig_n:] = 1
    real_ids = list(X.index)
    syn_ids = [f"SYN{i:05d}" for i in range(int(is_syn.sum()))]
    sample_ids = real_ids + syn_ids
    X_bal.insert(0, "Sample", sample_ids)

    y_bal = pd.Series(y_res, index=sample_ids, name="label")
    return X_bal, y_bal, kept_names, is_syn

#Cross-validation with scaling inside folds
def run_leakage_safe_cv(
    X_bal: pd.DataFrame,
    y_bal: pd.Series,
    kept_names: List[str],
    cv: int,
    random_state: int,
) -> pd.DataFrame:
    """Train baseline models with scaling after balancing (fit within each train fold only).
    """
    X = X_bal[kept_names].to_numpy(dtype=float)
    y = y_bal.to_numpy()

    cv = max(2, min(cv, int((y == y.min()).sum())))  # ensure minority present per fold
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    # Models
    models: Dict[str, object] = {}
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", solver="saga", random_state=random_state)
    models["LogisticRegression"] = lr

    lin = LinearSVC(C=0.5, tol=1e-3, max_iter=50000, dual=True, class_weight="balanced", random_state=random_state)
    models["LinearSVM"] = CalibratedClassifierCV(lin, method="sigmoid", cv=3)

    rf = RandomForestClassifier(n_estimators=400, random_state=random_state, class_weight="balanced_subsample", n_jobs=-1)
    models["RandomForest"] = rf

    knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
    models["kNN"] = knn

    gnb = GaussianNB()
    models["GaussianNB"] = gnb

    rows = []
    for mname, base in models.items():
        aucs = []; bals = []; f1s = []; precs = []; recs = []; accs = []
        for tr, te in skf.split(X, y):
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X[tr])
            Xte = scaler.transform(X[te])
            clf = clone(base)
            try:
                clf.fit(Xtr, y[tr])
            except Exception as e:
                print(f"[CV] Fit failed for {mname}: {e}")
                continue
            # AUC
            try:
                if hasattr(clf, "predict_proba"):
                    yscore = clf.predict_proba(Xte)[:, 1]
                else:
                    scores = np.asarray(clf.decision_function(Xte)).ravel()
                    smin, smax = scores.min(), scores.max()
                    yscore = (scores - smin) / (smax - smin + 1e-12)
                aucs.append(roc_auc_score(y[te], yscore))
            except Exception as e:
                print(f"[CV] AUC failed for {mname}: {e}")
            # Other metrics
            try:
                ypred = clf.predict(Xte)
                bals.append(balanced_accuracy_score(y[te], ypred))
                f1s.append(f1_score(y[te], ypred))
                precs.append(precision_score(y[te], ypred))
                recs.append(recall_score(y[te], ypred))
                accs.append(accuracy_score(y[te], ypred))
            except Exception as e:
                print(f"[CV] Metrics failed for {mname}: {e}")
        def ms(xs):
            xs = np.array(xs, dtype=float)
            return (float(np.nanmean(xs)) if xs.size else np.nan,
                    float(np.nanstd(xs)) if xs.size else np.nan)
        auc_m, auc_s = ms(aucs); bal_m, bal_s = ms(bals); f1_m, f1_s = ms(f1s)
        pr_m, pr_s = ms(precs); rc_m, rc_s = ms(recs); acc_m, acc_s = ms(accs)
        rows.append({
            "model": mname,
            "roc_auc_mean": auc_m, "roc_auc_std": auc_s,
            "bal_acc_mean": bal_m, "bal_acc_std": bal_s,
            "f1_mean": f1_m, "f1_std": f1_s,
            "precision_mean": pr_m, "precision_std": pr_s,
            "recall_mean": rc_m, "recall_std": rc_s,
            "accuracy_mean": acc_m, "accuracy_std": acc_s,
        })
    return pd.DataFrame(rows)

# Graphs orchestration 

def write_balanced_files_for_graphs(
    X_bal: pd.DataFrame,
    y_bal: pd.Series,
    is_syn: np.ndarray,
    out_dir: Path,
    condition_col_name: str,
    label_to_name: Optional[Dict[int, str]] = None,
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Feature table (rows=samples, cols=features) — drop helper columns
    feat = X_bal.drop(columns=[c for c in ["label", "is_synthetic"] if c in X_bal.columns]).set_index("Sample")
    feat.to_csv(out_dir / "otu_balanced.csv")

    # Metadata with condition column
    if label_to_name is None:
        label_to_name = {0: "Control", 1: "Case"}
    meta = pd.DataFrame({
        "Sample": y_bal.index,
        condition_col_name: [label_to_name.get(int(v), str(v)) for v in y_bal.values],
        "label": y_bal.values,
        "is_synthetic": is_syn,
    }).set_index("Sample")
    meta.to_csv(out_dir / "metadata_balanced.csv")

    return out_dir / "otu_balanced.csv", out_dir / "metadata_balanced.csv"

def run_graphs_cli(
    graphs_script: Path,
    otu_csv: Path,
    meta_csv: Path,
    graphs_out: Path,
    sample_id: str,
    condition_id: str,
    nboots: int,
    pval: float,
    min_samples: int,
    jobs: int,
    seed: int,
    pearson_threshold: float,
    pearson_topk: int,
    matrices_only: bool,
    no_pearson: bool,
    no_progress: bool,
) -> None:
    cmd = [
        sys.executable, str(graphs_script), "bootstrap",
        "--otu", str(otu_csv),
        "--metadata", str(meta_csv),
        "--out", str(graphs_out),
        "--sample-id", sample_id,
        "--condition-id", condition_id,
        "--nboots", str(nboots),
        "--pval", str(pval),
        "--min-samples", str(min_samples),
        "--jobs", str(jobs),
        "--pearson-threshold", str(pearson_threshold),
        "--pearson-topk", str(pearson_topk),
        "--seed", str(seed),
    ]
    if matrices_only:
        cmd.append("--matrices-only")
    if no_pearson:
        cmd.append("--no-pearson")
    if no_progress:
        cmd.append("--no-progress")

    print("[GRAPHS] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

# CLI
def main():
    p = argparse.ArgumentParser(description="Balance dataset in abundance space, save CSVs + (optional) graphs and CV metrics")
    p.add_argument("--feature-file", type=Path, required=True, help="OTU/species/genera table (rows=samples, cols=features)")
    p.add_argument("--meta-file", type=Path, required=True, help="Metadata file")
    p.add_argument("--id-col", type=str, default=None, help="Metadata column matching feature table index (e.g., 'Sample')")
    p.add_argument("--label-col", type=str, required=True, help="Metadata column to map into binary labels")
    p.add_argument("--positive", type=str, required=True, help="Comma-separated values considered positive (case-insensitive)")

    p.add_argument("--out-dir", type=Path, required=True, help="Output directory for balanced CSVs and optional results")
    p.add_argument("--sampler", type=str, default="AdaLoRAS", choices=["AdaLoRAS","SMOTE","NoSampling"], help="Balancing method")

    # VarianceThreshold (abundance space)
    p.add_argument("--variance-threshold", type=float, default=0.0)

    # AdaLoRAS params
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--k-all", type=int, default=5)
    p.add_argument("--k-min", type=int, default=20)
    p.add_argument("--n-shadows", type=int, default=50)
    p.add_argument("--affine-dim", type=int, default=5)
    p.add_argument("--noise-scale", type=float, default=0.15)

    p.add_argument("--random-state", type=int, default=RANDOM_STATE)

    # Optional CV
    p.add_argument("--cv", type=int, default=0, help="If >0, run leakage-safe CV with scaling in-fold on balanced data")
    p.add_argument("--cv-out", type=Path, default=None, help="Where to write CV results CSV (defaults under out-dir)")

    # Graphs
    p.add_argument("--run-graphs", action="store_true", help="If set, call your graphs script on the balanced dataset")
    p.add_argument("--graphs-script", type=Path, default=None, help="Path to your graph builder script (e.g., graphs.py)")
    p.add_argument("--graphs-out", type=Path, default=None, help="Where to put graph outputs (default: out-dir/graphs)")
    p.add_argument("--graphs-sample-id", type=str, default="Sample")
    p.add_argument("--graphs-condition-id", type=str, default="Study.Group")
    p.add_argument("--nboots", type=int, default=10)
    p.add_argument("--pval", type=float, default=0.05)
    p.add_argument("--min-samples", type=int, default=5)
    p.add_argument("--jobs", type=int, default=-1)
    p.add_argument("--pearson-threshold", type=float, default=0.3)
    p.add_argument("--pearson-topk", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--matrices-only", action="store_true")
    p.add_argument("--no-pearson", action="store_true")
    p.add_argument("--no-progress", action="store_true")

    args = p.parse_args()

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[STEP] Reading & aligning tables ...")
    X, meta = read_tables(args.feature_file, args.meta_file, id_col=args.id_col)
    y = infer_labels(meta, args.label_col, args.positive)
    X = X.loc[y.index]

    print("[INFO] Samples:", len(X), "Features:", X.shape[1])
    print("[INFO] Class counts:", y.value_counts().to_dict())

    print("[STEP] Building balanced dataset in abundance space (sampler=", args.sampler, ") ...", sep="")
    X_bal, y_bal, kept_names, is_syn = make_balanced_dataset(
        X=X, y=y, sampler_name=args.sampler,
        var_threshold=args.variance_threshold,
        random_state=args.random_state,
        beta=args.beta, k_all=args.k_all, k_min=args.k_min,
        n_shadows=args.n_shadows, affine_dim=args.affine_dim, noise_scale=args.noise_scale,
    )

    # Attach label + is_synthetic columns for convenience in a consolidated CSV too
    balanced_csv = outdir / f"balanced_{args.sampler.lower()}_abundance.csv"
    df_bal_out = X_bal.copy()
    df_bal_out["label"] = y_bal.values
    df_bal_out["is_synthetic"] = is_syn
    df_bal_out.to_csv(balanced_csv, index=False)
    print(f"[SAVE] Balanced dataset -> {balanced_csv}")

    # Optional CV
    if args.cv and args.cv > 0:
        print("[STEP] Running leakage-safe CV on balanced data ...")
        cv_df = run_leakage_safe_cv(X_bal=X_bal, y_bal=y_bal, kept_names=kept_names,
                                    cv=args.cv, random_state=args.random_state)
        cv_out = args.cv_out or (outdir / f"cv_results_{args.sampler.lower()}.csv")
        cv_df.to_csv(cv_out, index=False)
        print(f"[SAVE] CV results -> {cv_out}")

    # Graphs
    if args.run_graphs:
        if not args.graphs_script:
            raise SystemExit("--run-graphs was set but --graphs-script was not provided")
        graphs_out = args.graphs_out or (outdir / "graphs")
        print("[STEP] Preparing balanced files for graph builder ...")
        otu_csv, meta_csv = write_balanced_files_for_graphs(
            X_bal=df_bal_out, y_bal=y_bal, is_syn=is_syn, out_dir=outdir / "balanced_for_graphs",
            condition_col_name=args.graphs_condition_id,
        )
        print("[STEP] Calling graph builder ...")
        run_graphs_cli(
            graphs_script=args.graphs_script,
            otu_csv=otu_csv,
            meta_csv=meta_csv,
            graphs_out=graphs_out,
            sample_id=args.graphs_sample_id,
            condition_id=args.graphs_condition_id,
            nboots=args.nboots,
            pval=args.pval,
            min_samples=args.min_samples,
            jobs=args.jobs,
            seed=args.seed,
            pearson_threshold=args.pearson_threshold,
            pearson_topk=args.pearson_topk,
            matrices_only=args.matrices_only,
            no_pearson=args.no_pearson,
            no_progress=args.no_progress,
        )
        print(f"[DONE] Graph outputs at: {graphs_out}")

    print("[DONE] All tasks complete.")


if __name__ == "__main__":
    main()
