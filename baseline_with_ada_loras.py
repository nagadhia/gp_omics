#!/usr/bin/env python3
"""
Baseline classifiers on the normalized OTU/genera table, comparing No Sampling, AdaLoRAS, and SMOTE.
- Leakage-safe CV (preprocessing fit on train only).
- Saves AdaLoRAS-resampled TRAIN sets per fold/model (optional).
- Optionally saves ONE full-dataset oversampled CSV per sampler after CV.
- Uses ORIGINAL FEATURE NAMES (post-VarianceThreshold) in all saved CSVs.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Imbalanced-learn for SMOTE
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("Error: imbalanced-learn (imblearn) is required for SMOTE.")
    print("Please install it: pip install imbalanced-learn")
    import sys
    sys.exit(1)

from sklearn.base import BaseEstimator, ClassifierMixin, clone
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
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, accuracy_score
)
from sklearn.neighbors import NearestNeighbors

RANDOM_STATE = 42

# --------------------- AdaLoRAS Oversampler (with per-fold saving) ---------------------

class AdaLoRASOversampler:
    """Hybrid ADASYN + LoRAS oversampler, with optional per-fold saving."""
    def __init__(self, minority_class=1, beta=1.0, k_all=5, k_min=20,
                 n_shadows=50, affine_dim=5, noise_scale=0.15,
                 convex=True, feature_clip=None, random_state=None,
                 save_dir=None):
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
        self.save_dir = Path(save_dir) if save_dir else None
        # expected keys in _context: fold, model, sampling, orig_train_size, feature_names
        self._context = {}

    def _colnames_for(self, n_cols: int):
        names = self._context.get("feature_names", None)
        if isinstance(names, (list, tuple)) and len(names) == n_cols:
            return list(names)
        # fallback to numeric if not provided/mismatch
        return [f"f{i}" for i in range(n_cols)]

    def _maybe_save(self, X_res, y_res):
        if not self.save_dir:
            return
        fold = self._context.get("fold", "NA")
        model = str(self._context.get("model", "UnknownModel")).replace("/", "-")
        sampling = str(self._context.get("sampling", "AdaLoRAS"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{sampling}__{model}__fold-{fold}.csv"
        out_path = self.save_dir / fname

        colnames = self._colnames_for(X_res.shape[1])
        df = pd.DataFrame(X_res, columns=colnames)
        df["label"] = y_res
        orig_n = int(self._context.get("orig_train_size", 0))
        is_syn = np.zeros(len(y_res), dtype=int)
        if orig_n > 0 and len(y_res) >= orig_n:
            is_syn[orig_n:] = 1
        df["is_synthetic"] = is_syn
        df.to_csv(out_path, index=False)
        print(f"[AdaLoRAS] Saved oversampled TRAIN set for fold {fold}, model {model} -> {out_path}")

    def fit_resample(self, X, y):
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        X_min = X[y == self.minority_class]
        X_maj = X[y != self.minority_class]
        n_min, n_maj = len(X_min), len(X_maj)

        def _return_and_maybe_save(Xr, yr):
            if "orig_train_size" not in self._context:
                self._context["orig_train_size"] = len(yr)
            self._maybe_save(Xr, yr)
            return Xr, yr

        if n_min == 0 or n_maj == 0 or self.beta <= 0:
            return _return_and_maybe_save(X.copy(), y.copy())

        gap = n_maj - n_min
        if gap <= 0:
            return _return_and_maybe_save(X.copy(), y.copy())

        G = int(np.floor(gap * self.beta))
        if G == 0:
            return _return_and_maybe_save(X.copy(), y.copy())

        k_all_eff = max(1, min(self.k_all, len(X) - 1))
        knn_all = NearestNeighbors(n_neighbors=k_all_eff).fit(X)

        k_min_eff = max(1, min(self.k_min, max(1, n_min - 1)))
        knn_min = NearestNeighbors(n_neighbors=k_min_eff).fit(X_min)

        q25, q75 = np.percentile(X_min, [25, 75], axis=0)
        iqr = np.maximum(q75 - q25, 1e-12)
        shadow_sigma = self.noise_scale * iqr

        fmin = fmax = lo = hi = None
        if self.feature_clip == "minmax":
            fmin = X.min(axis=0)
            fmax = X.max(axis=0)
        elif isinstance(self.feature_clip, (tuple, list)) and len(self.feature_clip) == 2:
            lo, hi = self.feature_clip

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

        synth = []
        min_neighbor_cache = {}
        for i, xi in enumerate(X_min):
            _, idxs = knn_min.kneighbors(xi.reshape(1, -1))
            min_neighbor_cache[i] = X_min[idxs[0]]

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
            return _return_and_maybe_save(X.copy(), y.copy())

        X_synth = np.vstack(synth)
        y_synth = np.full(len(synth), self.minority_class, dtype=y.dtype)

        X_res = np.vstack([X, X_synth])
        y_res = np.concatenate([y, y_synth])

        self._maybe_save(X_res, y_res)
        return X_res, y_res

# --------------------- ResampledPipeline ---------------------

class ResampledPipeline(BaseEstimator, ClassifierMixin):
    """Preprocess -> (optional) oversample -> fit classifier, inside CV folds."""
    _estimator_type = "classifier"

    def __init__(self, preprocessor, classifier, sampler=None):
        self.preprocessor = preprocessor
        self.classifier = classifier
        self.sampler = sampler

    def fit(self, X, y):
        # Keep original feature names if present (pandas DataFrame)
        if hasattr(X, "columns"):
            orig_cols = list(X.columns)
        else:
            orig_cols = [f"f{i}" for i in range(np.array(X).shape[1])]

        self.preprocessor_ = clone(self.preprocessor)
        Xp = self.preprocessor_.fit_transform(X, y)

        # Extract kept feature names after VarianceThreshold
        kept_cols = None
        try:
            vt = self.preprocessor_.named_steps.get("var", None)
            if vt is not None and hasattr(vt, "get_support"):
                mask = vt.get_support()
                kept_cols = [c for c, keep in zip(orig_cols, mask) if keep]
        except Exception:
            kept_cols = None

        if self.sampler is not None:
            # Pass feature names to the sampler so saved CSVs use original names
            if hasattr(self.sampler, "_context"):
                if kept_cols is not None:
                    self.sampler._context["feature_names"] = kept_cols
                if "orig_train_size" not in self.sampler._context:
                    self.sampler._context["orig_train_size"] = len(y)
            X_fit, y_fit = self.sampler.fit_resample(Xp, y)
        else:
            X_fit, y_fit = Xp, y

        self.classifier_ = clone(self.classifier)
        self.classifier_.fit(X_fit, y_fit)
        self.classes_ = np.unique(y_fit)
        return self

    def predict(self, X):
        Xp = self.preprocessor_.transform(X)
        return self.classifier_.predict(Xp)

    def predict_proba(self, X):
        Xp = self.preprocessor_.transform(X)
        if hasattr(self.classifier_, "predict_proba"):
            return self.classifier_.predict_proba(Xp)
        if hasattr(self.classifier_, "decision_function"):
            scores = np.asarray(self.classifier_.decision_function(Xp)).ravel()
            smin, smax = scores.min(), scores.max()
            probs1 = (scores - smin) / (smax - smin + 1e-12)
            return np.column_stack([1 - probs1, probs1])
        raise AttributeError("Base classifier has neither predict_proba nor decision_function.")

# --- Utility Functions ---

def _read_any_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    ext = path.suffix.lower()
    sep = "\t" if ext in {".tsv", ".txt"} else ","
    return pd.read_csv(path, index_col=0, sep=sep)

def read_tables(feature_file: Path, meta_file: Path, id_col: str | None = None):
    """Read feature + metadata tables and align by sample ID."""
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

    if len(X.index.intersection(meta.index.astype(str))) == 0:
        candidates = [c for c in ["Sample", "sample", "SampleID", "Subject", "ID"] if c in meta.columns]
        best_col, best_overlap = None, 0
        for c in candidates:
            vals = meta[c].astype(str).str.strip()
            overlap = len(set(X.index).intersection(set(vals)))
            if overlap > best_overlap:
                best_overlap, best_col = overlap, c
        if best_col is not None and best_overlap > 0:
            meta[best_col] = meta[best_col].astype(str).str.strip()
            meta = meta.set_index(best_col, drop=False)

    common = X.index.intersection(meta.index.astype(str))
    if len(common) == 0:
        raise SystemExit(
            "No overlapping sample IDs between features and metadata after alignment. "
            "Use --id-col to specify the metadata ID column (e.g., 'Sample')."
        )

    X = X.loc[common].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    meta = meta.loc[common]
    return X, meta

def infer_labels(meta: pd.DataFrame, label_col: str, positive: str):
    if label_col not in meta.columns:
        raise SystemExit(f"--label-col '{label_col}' not found in metadata columns: {list(meta.columns)}")
    y_raw = meta[label_col].astype(str).str.strip()
    pos_values = {p.strip().lower() for p in positive.split(",")}
    y = y_raw.str.lower().isin(pos_values).astype(int)
    if y.nunique() != 2:
        raise SystemExit(f"Labels not binary after mapping, got {y.value_counts().to_dict()}")
    return y

# --------------------- Models ---------------------

def build_models(sampler, var_thresh: float):
    vt = ("var", VarianceThreshold(threshold=var_thresh))
    scaler = ("scale", StandardScaler())
    preproc = Pipeline([vt, scaler])

    models = {}

    lr = LogisticRegression(max_iter=2000, class_weight="balanced", solver="saga", random_state=RANDOM_STATE)
    models["LogisticRegression"] = ResampledPipeline(preprocessor=preproc, classifier=lr, sampler=sampler)

    lin_svc = LinearSVC(
        C=0.5, tol=1e-3, max_iter=50000, dual=True, class_weight="balanced", random_state=RANDOM_STATE
    )
    svm = CalibratedClassifierCV(lin_svc, method="sigmoid", cv=3)
    models["LinearSVM"] = ResampledPipeline(preprocessor=preproc, classifier=svm, sampler=sampler)

    rf = RandomForestClassifier(
        n_estimators=400, random_state=RANDOM_STATE, class_weight="balanced_subsample", n_jobs=-1
    )
    models["RandomForest"] = ResampledPipeline(preprocessor=preproc, classifier=rf, sampler=sampler)

    knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
    models["kNN"] = ResampledPipeline(preprocessor=preproc, classifier=knn, sampler=sampler)

    gnb = GaussianNB()
    models["GaussianNB"] = ResampledPipeline(preprocessor=preproc, classifier=gnb, sampler=sampler)

    return models

# --------------------- Main ---------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-file", type=Path, required=True, help="Path to OTU/genera feature table (.csv/.tsv/.txt)")
    ap.add_argument("--meta-file", type=Path, required=True, help="Path to metadata table (.csv/.tsv/.txt)")
    ap.add_argument("--id-col", type=str, default=None,
                    help="Column in metadata containing sample IDs to align with feature rows (e.g., 'Sample')")
    ap.add_argument("--label-col", required=True, help="Column in metadata with labels")
    ap.add_argument("--positive", required=True, help="Comma-separated positives for disease class (case-insensitive)")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--out-csv", type=Path, default=Path("otu_sampling_comparison_results.csv"))

    # preprocessing
    ap.add_argument("--variance-threshold", type=float, default=0.0, help="Per-fold VarianceThreshold")

    # oversampling parameters (for AdaLoRAS)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--k-all", type=int, default=5)
    ap.add_argument("--k-min", type=int, default=20)
    ap.add_argument("--n-shadows", type=int, default=50)
    ap.add_argument("--affine-dim", type=int, default=5)
    ap.add_argument("--noise-scale", type=float, default=0.15)
    ap.add_argument("--feature-clip", type=str, default="minmax")
    ap.add_argument("--random-state", type=int, default=RANDOM_STATE)

    # Save per-fold AdaLoRAS resampled training sets
    ap.add_argument("--save-resampled-dir", type=Path, default=None,
                    help="Directory to save AdaLoRAS-resampled TRAIN sets for each fold and model")

    # Save ONE full oversampled dataset per sampler (after CV)
    ap.add_argument("--save-full-dir", type=Path, default=None,
                    help="If set, save a single full-dataset oversampled CSV per sampler (preprocessed space).")
    ap.add_argument("--save-full-samplers", type=str, default="AdaLoRAS,SMOTE",
                    help="Comma-separated list of samplers to export: e.g., 'AdaLoRAS', 'SMOTE', 'No Sampling'")

    args = ap.parse_args()

    print("--- Reading Data and Inferring Labels ---")
    X, meta = read_tables(args.feature_file, args.meta_file, id_col=args.id_col)
    y = infer_labels(meta, args.label_col, args.positive)
    X = X.loc[y.index]  # keep alignment

    minority_class = y.value_counts().idxmin()
    n_min = int((y == minority_class).sum())
    n_maj = int((y != minority_class).sum())
    print(f"Total samples: {len(X)}. Minority class ({minority_class}): {n_min}. Majority class: {n_maj}.")

    # Parse feature clipping option
    feature_clip = None
    if isinstance(args.feature_clip, str):
        if args.feature_clip.lower() == "minmax":
            feature_clip = "minmax"
        elif args.feature_clip.lower() != "none":
            if "," in args.feature_clip:
                lo, hi = [float(t) for t in args.feature_clip.split(",")]
                feature_clip = (lo, hi)

    # Oversampler params
    overs_params = dict(
        minority_class=minority_class,
        beta=args.beta,
        k_all=args.k_all,
        k_min=args.k_min,
        n_shadows=args.n_shadows,
        affine_dim=args.affine_dim,
        noise_scale=args.noise_scale,
        convex=True,
        feature_clip=feature_clip,
        random_state=args.random_state,
        save_dir=args.save_resampled_dir,
    )

    # Sampling strategies
    sampling_strategies = {
        "No Sampling": None,
        "AdaLoRAS": AdaLoRASOversampler(**overs_params),
    }
    if n_min > 1:
        k_smote = max(1, min(5, n_min - 1))
        sampling_strategies["SMOTE"] = SMOTE(k_neighbors=k_smote, random_state=args.random_state)
    else:
        print("[Warn] Skipping SMOTE: minority class has <=1 sample.")

    # CV setup (ensure minority in each fold)
    effective_cv = min(args.cv, max(2, n_min))
    if effective_cv != args.cv:
        print(f"[Warn] Reducing CV from {args.cv} to {effective_cv} to keep minority class in each fold.")
    cv = StratifiedKFold(n_splits=effective_cv, shuffle=True, random_state=args.random_state)

    rows = []
    print("\n--- Running Cross-Validation ---")
    for sampling_name, sampler in sampling_strategies.items():
        print(f"-> Strategy: {sampling_name}")
        models = build_models(sampler=sampler, var_thresh=args.variance_threshold)

        for model_name, est in models.items():
            print(f"   - Model: {model_name}")
            aucs, bals, f1s, precs, recs, accs = [], [], [], [], [], []

            fold_no = 0
            for train_idx, test_idx in cv.split(X.values, y.values):
                fold_no += 1
                # Keep DataFrame to preserve column names!
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.values[train_idx], y.values[test_idx]

                est_fold = clone(est)

                # If AdaLoRAS is used, set context for saving per-fold TRAIN data
                if sampling_name == "AdaLoRAS" and est_fold.sampler is not None:
                    est_fold.sampler._context = {
                        "fold": fold_no,
                        "model": model_name,
                        "sampling": sampling_name,
                        "orig_train_size": len(y_train),
                        # feature_names will be filled inside ResampledPipeline.fit
                    }

                # Fit
                try:
                    est_fold.fit(X_train, y_train)
                except Exception as e:
                    print(f"      [Error] Fit failed on fold {fold_no}: {e}")
                    continue

                # AUC
                try:
                    proba = est_fold.predict_proba(X_test)
                    y_score = proba[:, 1] if proba.ndim > 1 else proba
                    aucs.append(roc_auc_score(y_test, y_score))
                except Exception as e:
                    print(f"      [Warn] AUC failed on fold {fold_no}: {e}")

                # Other metrics
                try:
                    y_pred = est_fold.predict(X_test)
                    bals.append(balanced_accuracy_score(y_test, y_pred))
                    f1s.append(f1_score(y_test, y_pred))
                    precs.append(precision_score(y_test, y_pred))
                    recs.append(recall_score(y_test, y_pred))
                    accs.append(accuracy_score(y_test, y_pred))
                except Exception as e:
                    print(f"      [Warn] Metric failed on fold {fold_no}: {e}")

            # Aggregate
            def mean_std(xs):
                xs = np.array(xs, dtype=float)
                return (float(np.nanmean(xs)) if xs.size > 0 else np.nan,
                        float(np.nanstd(xs)) if xs.size > 0 else np.nan)

            auc_m, auc_s = mean_std(aucs)
            bal_m, bal_s = mean_std(bals)
            f1_m, f1_s = mean_std(f1s)
            pr_m, pr_s = mean_std(precs)
            rc_m, rc_s = mean_std(recs)
            acc_m, acc_s = mean_std(accs)

            rows.append({
                "sampling_strategy": sampling_name,
                "model": model_name,
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "roc_auc_mean": auc_m, "roc_auc_std": auc_s,
                "bal_acc_mean": bal_m, "bal_acc_std": bal_s,
                "f1_mean": f1_m, "f1_std": f1_s,
                "precision_mean": pr_m, "precision_std": pr_s,
                "recall_mean": rc_m, "recall_std": rc_s,
                "accuracy_mean": acc_m, "accuracy_std": acc_s,
            })

    # --- Optional: save ONE full-dataset oversampled CSV per sampler ---
    if args.save_full_dir is not None:
        save_dir = Path(args.save_full_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        vt = VarianceThreshold(threshold=args.variance_threshold)
        scaler = StandardScaler()
        full_preproc = Pipeline([("var", vt), ("scale", scaler)])
        Xp_full = full_preproc.fit_transform(X, y.values)

        # ORIGINAL kept feature names after VT
        kept_names = list(X.columns)
        try:
            mask_full = full_preproc.named_steps["var"].get_support()
            kept_names = [c for c, keep in zip(list(X.columns), mask_full) if keep]
        except Exception:
            pass

        def _dump_resampled(name, X_res, y_res, orig_n, out_dir, colnames):
            if len(colnames) != X_res.shape[1]:
                colnames = [f"f{i}" for i in range(X_res.shape[1])]
            df_res = pd.DataFrame(X_res, columns=colnames)
            df_res["label"] = y_res
            is_syn = np.zeros(len(y_res), dtype=int)
            is_syn[orig_n:] = 1
            df_res["is_synthetic"] = is_syn
            out_path = out_dir / f"{name}__full_preprocessed.csv"
            df_res.to_csv(out_path, index=False)
            print(f"[SAVE] Wrote full oversampled dataset -> {out_path}")

        want = {s.strip() for s in args.save_full_samplers.split(",") if s.strip()}

        # No Sampling
        if "No Sampling" in want or "NoSampling" in want:
            _dump_resampled("NoSampling", Xp_full, y.values.copy(), orig_n=len(y), out_dir=save_dir, colnames=kept_names)

        # AdaLoRAS
        if "AdaLoRAS" in want:
            adaloras_full = AdaLoRASOversampler(
                minority_class=y.value_counts().idxmin(),
                beta=args.beta, k_all=args.k_all, k_min=args.k_min,
                n_shadows=args.n_shadows, affine_dim=args.affine_dim,
                noise_scale=args.noise_scale, convex=True,
                feature_clip=("minmax" if str(args.feature_clip).lower()=="minmax" else None),
                random_state=args.random_state,
                save_dir=None
            )
            X_res, y_res = adaloras_full.fit_resample(Xp_full, y.values)
            _dump_resampled("AdaLoRAS", X_res, y_res, orig_n=len(y), out_dir=save_dir, colnames=kept_names)

        # SMOTE
        if "SMOTE" in want:
            n_min_full = int((y.values == y.value_counts().idxmin()).sum())
            if n_min_full > 1:
                k_smote = max(1, min(5, n_min_full - 1))
                smote_full = SMOTE(k_neighbors=k_smote, random_state=args.random_state)
                X_res, y_res = smote_full.fit_resample(Xp_full, y.values)
                _dump_resampled("SMOTE", X_res, y_res, orig_n=len(y), out_dir=save_dir, colnames=kept_names)
            else:
                print("[SAVE] Skipping SMOTE full export: minority class has <=1 sample.")

    # --- Output Final Table ---
    df = pd.DataFrame(rows)
    metric_means = [c for c in df.columns if c.endswith("_mean")]
    metric_stds  = [m.replace("_mean", "_std") for m in metric_means]
    order = ["sampling_strategy", "model", "n_samples", "n_features"]
    for m_mean, m_std in zip(metric_means, metric_stds):
        if m_mean in df.columns: order.append(m_mean)
        if m_std  in df.columns: order.append(m_std)
    df_out = df.reindex(columns=order)
    df_out = df_out.sort_values(by=["model", "roc_auc_mean"], ascending=[True, False])

    out_path = args.out_csv
    df_out.to_csv(out_path, index=False)
    print(f"\n--- Final Results saved to {out_path} ---")
    print(df_out.to_string(index=False))

if __name__ == "__main__":
    main()

python balance_with_graphs.py   --feature-file microbiome-metabolome-curated-data/data/processed_data/FRANZOSA_IBD_2019/genera.tsv   --meta-file microbiome-metabolome-curated-data/data/processed_data/FRANZOSA_IBD_2019/metadata.tsv   --id-col Sample   --label-col Study.Group   --positive "control"   --variance-threshold 0   --sampler AdaLoRAS   --out-dir microbiome-metabolome-curated-data/data/processed_data/FRANZOSA_IBD_2019/balanced_abundance   --run-graphs   --graphs-script graphs.py   --graphs-out microbiome-metabolome-curated-data/data/processed_data/FRANZOSA_IBD_2019/graphs_balanced_abundance   --graphs-sample-id Sample   --graphs-condition-id Study.Group