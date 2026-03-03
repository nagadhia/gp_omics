#!/usr/bin/env python3
"""
Build per-sample species/genera(feature) graphs for the curated Borenstein-lab repo. This can actually be any repo but it needs to have the structure:
     
      Processed_Data/
        feature_table_normalized.csv
        metadata_aligned.csv 
        
        
For each sample, we write two feature-feature graphs::
  1) Pearson LIONESS graph  (https://doi.org/10.1016/j.isci.2019.03.021)
  2) Weighted bootstrapped graph (https://doi.org/10.1145/3584371.3612969)

Works in two modes:
  * Repo mode:  --repo-root  --dataset  [--prefer-species]
  * File mode:  --otu  --metadata  (rows=samples, cols=features)

Outputs (bootstrap mode):
  OUTDIR/Bootstrap_Filtering_(P_<p>_N_<n>)/
    ├─ Condition_Summaries/
    │    ├─ filtering_summary_<condition>.csv
    │    ├─ feature_table_filtered_<condition>.csv
    │    ├─ boot_summary_mean.csv                 (CSV instead of NPZ)
    │    └─ boot_summary_std.csv                  (CSV instead of NPZ)
    └─ Individual_Samples_by_Condition/<condition>/<sample>/
         ├─ graph_bootstrap_filtered.graphml            (optional; omitted with --matrices-only)
         ├─ weights_bootstrap_filtered.csv              (CSV instead of NPZ; W matrix)
         ├─ graph_pearson_lioness.graphml               (optional; omitted with --matrices-only)
         └─ weights_pearson_lioness.csv                 (CSV instead of NPZ; R matrix)

Speed & UX:
- Fast LIONESS via sufficient statistics (with progress)
- Memory-safe bootstrapping via online mean/std (Welford) with progress
- Joblib parallel over samples and LIONESS; avoid BLAS oversubscription in shell
- Progress bars can be disabled with --no-progress
"""

from __future__ import annotations

import warnings
warnings.filterwarnings(
    "ignore",
    message="networkx backend defined more than once",
    category=RuntimeWarning,
)

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import joblib
from joblib import Parallel, delayed
from scipy import stats
from tqdm.auto import tqdm
from contextlib import contextmanager

# Thresholds
EPS_PRES: float = 1e-9     # presence threshold for a feature to be considered "present" in a sample
EPS_DIV: float  = 1e-12    # epsilon to avoid division by zero
WEIGHT_CLIP: float = 1e6   # cap extreme weights to avoid numerical explosions


# Progress helpers

@contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar.
    Advances on batch completion (actual task completion), not just submission.
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_cb = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_cb
        tqdm_object.close()


# I/O helpers 

def load_repo_feature_and_metadata(repo_root: Path, dataset: str, prefer_species: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (feature_table, metadata) aligned by Sample ID.
    feature_table: rows=samples, cols=features (floats)
    metadata: index=Sample (str)
    """
    ds_root = Path(repo_root) / "data" / "processed_data" / dataset
    if not ds_root.exists():
        raise SystemExit(f"Dataset path not found: {ds_root}")

    # Load microbe table (species preferred by default)
    ft_fp = ds_root / ("species.tsv" if prefer_species else "genera.tsv")
    if not ft_fp.exists():
        alt = ds_root / ("genera.tsv" if prefer_species else "species.tsv")
        ft_fp = alt if alt.exists() else None
    if ft_fp is None or not ft_fp.exists():
        raise SystemExit(f"No species.tsv or genera.tsv found in {ds_root}")

    feature_table = pd.read_csv(ft_fp, sep="\t", index_col=0)
    feature_table = feature_table.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    feature_table.index = feature_table.index.astype(str)  # rows are samples in this repo

    # Load metadata and index by Sample
    meta_fp = ds_root / "metadata.tsv"
    if not meta_fp.exists():
        raise SystemExit(f"metadata.tsv not found in {ds_root}")
    metadata = pd.read_csv(meta_fp, sep="\t")
    if "Sample" not in metadata.columns:
        raise SystemExit(f"'Sample' column not found in {meta_fp}. Columns: {list(metadata.columns)}")
    metadata["Sample"] = metadata["Sample"].astype(str)
    metadata = metadata.drop_duplicates(subset=["Sample"], keep="first").set_index("Sample")

    # Align
    common = feature_table.index.intersection(metadata.index.astype(str))
    if len(common) == 0:
        raise SystemExit("No common samples between feature table and metadata.")
    feature_table = feature_table.loc[common]
    metadata = metadata.loc[common]
    return feature_table, metadata


def process_data_generic(otu_file_path: Path, metadata_file_path: Path, output_dir: Path, sample_id: str = 'Sample', condition_id: str = 'Study.Group') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """File mode loader + light cleaning + relative normalization.
    Returns feature_table (rows=samples) and metadata (index=Sample).
    """
    processed_data_dir = output_dir / "01_Processed_Data"
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    sep_otu = '\t' if str(otu_file_path).lower().endswith('.tsv') else ','
    sep_meta = '\t' if str(metadata_file_path).lower().endswith('.tsv') else ','

    otu = pd.read_csv(otu_file_path, index_col=0, sep=sep_otu)
    meta = pd.read_csv(metadata_file_path, sep=sep_meta)

    if sample_id not in meta.columns:
        raise SystemExit(f"Sample ID column '{sample_id}' not in metadata. Columns: {meta.columns.tolist()}")
    if condition_id not in meta.columns:
        raise SystemExit(f"Condition ID column '{condition_id}' not in metadata. Columns: {meta.columns.tolist()}")

    meta[sample_id] = meta[sample_id].astype(str)
    meta = meta.drop_duplicates(subset=[sample_id], keep='first').set_index(sample_id)
    otu.index = otu.index.astype(str)

    common = otu.index.intersection(meta.index)
    if len(common) == 0:
        raise SystemExit("No common samples between OTU table and metadata.")

    otu = otu.loc[common].astype(float)
    meta = meta.loc[common]

    # drop all-zero features and zero-sum samples
    otu = otu.loc[:, (otu > 0).any(axis=0)]
    sums = otu.sum(axis=1)
    valid = sums > 0
    otu = otu.loc[valid]
    meta = meta.loc[valid]
    sums = sums.loc[valid]

    # normalize to relative abundance
    otu = otu.div(sums, axis=0).fillna(0.0)

    # Save aligned for inspection
    otu.to_csv(processed_data_dir / "feature_table_normalized.csv")
    meta.to_csv(processed_data_dir / "metadata_aligned.csv")
    print(f"[INFO] Wrote aligned files to: {processed_data_dir}")

    return otu, meta

# LIONESS helpers
def compute_feature_corr_lioness_fast(df_cond: pd.DataFrame, jobs: int = -1, progress: bool = True) -> Dict[str, np.ndarray]:
    """Per-sample feature-feature Pearson LIONESS matrices.
    df_cond: rows=samples, cols=features
    Returns: {sample_id -> matrix (F x F)}
    """
    # Accumulate in float64 for stable sums/variances; cast back at the end.
    X = df_cond.to_numpy(dtype=np.float64)  # n x f
    n, f = X.shape
    if n < 3 or f < 2:
        raise ValueError("Not enough samples/features for correlation.")

    S = X.sum(axis=0)
    S2 = (X**2).sum(axis=0)
    Sxy = X.T @ X
    mu_all = S / n
    cx_all = np.maximum(S2 - n * (mu_all**2), EPS_DIV)
    num_all = Sxy - n * np.outer(mu_all, mu_all)
    denom_all = np.sqrt(np.outer(cx_all, cx_all))
    with np.errstate(divide="ignore", invalid="ignore"):
        r_all = num_all / denom_all
    np.fill_diagonal(r_all, 1.0)
    np.clip(r_all, -1.0, 1.0, out=r_all)

    def one_minus(t: int):
        xt = X[t, :]
        S_m, S2_m = S - xt, S2 - xt**2
        Sxy_m = Sxy - np.outer(xt, xt)
        n_m = n - 1
        mu = S_m / n_m
        cx = np.maximum(S2_m - n_m * (mu**2), EPS_DIV)
        num = Sxy_m - n_m * np.outer(mu, mu)
        denom = np.sqrt(np.outer(cx, cx))
        with np.errstate(divide="ignore", invalid="ignore"):
            r_minus = num / denom
        np.fill_diagonal(r_minus, 1.0)
        np.clip(r_minus, -1.0, 1.0, out=r_minus)
        Es = n * r_all - (n - 1) * r_minus
        return df_cond.index[t], Es.astype(np.float32, copy=False)

    if progress:
        with tqdm_joblib(tqdm(total=n, desc="LIONESS per-sample", leave=False)):
            results = Parallel(n_jobs=jobs)(delayed(one_minus)(t) for t in range(n))
    else:
        results = Parallel(n_jobs=jobs)(delayed(one_minus)(t) for t in range(n))

    return {sid: Es for sid, Es in results}


# Bootstrapped weighted rule  + filtering

def compute_sample_weights(sample_rel_otu: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(sample_rel_otu, dtype=np.float32)
    n = arr.size
    binary = (arr > EPS_PRES).astype(np.int8)
    cooc = np.outer(binary, binary)

    a_i = arr.reshape(-1, 1)
    a_j = arr.reshape(1, -1)

    # symmetric weight: 2 a_i a_j / (a_i^2 + a_j^2)
    denom = a_i**2 + a_j**2
    with np.errstate(divide='ignore', invalid='ignore'):
        W = (2.0 * a_i * a_j) / denom
        np.nan_to_num(W, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # zero diagonal, clip extremes if you still want that behavior
    np.fill_diagonal(W, 0.0)
    np.clip(W, -WEIGHT_CLIP, WEIGHT_CLIP, out=W)
    return W.astype(np.float32), np.outer(binary, binary)

def compute_all_weights(otu_subset: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Average weights over a set of samples (used inside bootstraps)."""
    X = np.asarray(otu_subset, dtype=np.float32)
    n_samp, n_feat = X.shape
    Wsum = np.zeros((n_feat, n_feat), dtype=np.float32)
    Csum = np.zeros((n_feat, n_feat), dtype=np.float32)
    for i in range(n_samp):
        w, c = compute_sample_weights(X[i, :])
        if w.shape == Wsum.shape:
            Wsum += w
            Csum += c
    with np.errstate(divide='ignore', invalid='ignore'):
        avg = np.divide(Wsum, Csum, where=Csum != 0, out=np.zeros_like(Wsum))
    avg[~np.isfinite(avg)] = 0.0
    return avg, Csum


def bootstrap_condition_matrices(
    df_cond: pd.DataFrame,
    n_boots: int,
    out_dir: Path,
    jobs: int = -1,  # kept for compatibility; not used here
    seed: int = 42,
    save_boot_mats: bool = False,
    progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Path]:
    """Compute bootstrap mean and std of weight matrices without stacking all bootstraps in RAM.
    Uses Welford's algorithm; optionally saves each bootstrap matrix as CSV (compressed NPZ removed).
    Returns (mean, std, out_dir).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mats_dir = out_dir / "matrices"
    mats_dir.mkdir(parents=True, exist_ok=True)

    X = df_cond.to_numpy(dtype=np.float32)
    n_samples, n_feat = X.shape
    if n_samples == 0 or n_feat < 2:
        raise ValueError("Not enough data for bootstrapping.")

    rng = np.random.default_rng(seed)

    # Online mean/variance (Welford)
    mean = np.zeros((n_feat, n_feat), dtype=np.float32)
    m2   = np.zeros_like(mean)
    k = 0

    iterator = range(n_boots)
    if progress:
        iterator = tqdm(iterator, total=n_boots, desc="Bootstraps", leave=False)

    for b in iterator:
        idxs = rng.choice(n_samples, size=n_samples, replace=True)
        w, _ = compute_all_weights(X[idxs])
        w = w.astype(np.float32, copy=False)
        if save_boot_mats:
            # Save each bootstrap matrix as CSV
            np.savetxt(mats_dir / f"bstrap_weight_matrix_{b}.csv", w, delimiter=",", fmt="%.7g")
        k1 = k + 1
        delta = w - mean
        mean += delta / k1
        m2   += delta * (w - mean)
        k = k1

    if progress and hasattr(iterator, "close"):
        iterator.close()

    std = np.sqrt(m2 / max(k - 1, 1)).astype(np.float32, copy=False)
    # Save summaries for inspection (CSV)
    np.savetxt(out_dir / "boot_summary_mean.csv", mean, delimiter=",", fmt="%.7g")
    np.savetxt(out_dir / "boot_summary_std.csv",  std,  delimiter=",", fmt="%.7g")
    return mean, std, out_dir


def filter_sample_graphs_with_bootstrap(
    df_cond: pd.DataFrame,
    boot_means: np.ndarray,
    boot_stds: np.ndarray,
    out_base: Path,
    condition_group: str,
    p_thresh: float = 0.05,
    pearson_edges_per_sample: Optional[Dict[str, np.ndarray]] = None,
    pearson_threshold: float = 0.3,
    pearson_topk: Optional[int] = 10000,
    progress: bool = True,
    write_graphml: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build per-sample graphs for a condition using bootstrap summaries to filter edges.

    Filtering policy (consensus): keep edges that are NOT significantly different from the bootstrap mean i.e. 
    remove if p < p_thresh  (i.e., outliers vs the bootstrap distribution)

    """
    out_cond_samples = out_base / "Individual_Samples_by_Condition" / condition_group
    out_cond_samples.mkdir(parents=True, exist_ok=True)
    out_cond_summary = out_base / "Condition_Summaries"
    out_cond_summary.mkdir(parents=True, exist_ok=True)

    X = df_cond.to_numpy(dtype=np.float32)
    n_samples, n_feat = X.shape
    species = df_cond.columns

    means = np.asarray(boot_means, dtype=np.float32)
    stds  = np.asarray(boot_stds,  dtype=np.float32)

    def gaussian_pvals_vs_matrix(Ws: np.ndarray) -> np.ndarray:
        # Two-sided z-test using bootstrap std as scale (no /sqrt(n): comparing a single observed matrix to bootstrap spread)
        with np.errstate(divide='ignore', invalid='ignore'):
            scale = np.maximum(stds, EPS_DIV)
            z = (Ws - means) / scale
        p = 2.0 * stats.norm.sf(np.abs(z))
        p[~np.isfinite(p)] = 1.0
        return p

    summary_rows = {}
    filtered_otu = df_cond.copy()

    iu, ju = np.triu_indices(n_feat, k=1)

    iterator = range(n_samples)
    if progress:
        iterator = tqdm(iterator, total=n_samples, desc=f"Samples ({condition_group})", leave=False)

    for s_idx in iterator:
        sname = df_cond.index[s_idx]
        abund = X[s_idx, :]

        # Bootstrapped weighted graph (matrix)
        W_boot, _ = compute_sample_weights(abund)
        pvals = gaussian_pvals_vs_matrix(W_boot)
        # CONSENSUS: drop edges that significantly deviate from bootstrap mean
        mask_remove = (pvals < p_thresh) & np.isfinite(W_boot)
        W_boot_f = W_boot.copy()
        W_boot_f[mask_remove] = 0.0

        # Paths & persists
        s_out = out_cond_samples / sname
        s_out.mkdir(parents=True, exist_ok=True)
        # Always write the filtered matrix (dense) as CSV
        np.savetxt(s_out / "weights_bootstrap_filtered.csv", W_boot_f.astype(np.float32, copy=False), delimiter=",", fmt="%.7g")

        # Node presence bookkeeping (used for summaries and optional GraphML)
        present = np.where(abund > EPS_PRES)[0]
        present_set = {str(species[i]) for i in present}
        present_mask = (abund > EPS_PRES)

        if write_graphml:
            # Build NetworkX graph for bootstrap-filtered weights
            G_boot = nx.Graph()
            for i in present:
                node = str(species[i])
                G_boot.add_node(node, relab=float(abund[i]))
            for i, j in zip(iu, ju):
                w = float(W_boot_f[i, j])
                if w != 0.0 and np.isfinite(w):
                    ni, nj = str(species[i]), str(species[j])
                    if ni in present_set and nj in present_set:
                        G_boot.add_edge(ni, nj, weight=w)
            nx.write_graphml(G_boot, s_out / "graph_bootstrap_filtered.graphml")

        # Pearson LIONESS graph 
        edges_kept_p = 0
        if pearson_edges_per_sample is not None and sname in pearson_edges_per_sample:
            R_s = pearson_edges_per_sample[sname]
            # Always store full matrix for downstream usage as CSV
            np.savetxt(s_out / "weights_pearson_lioness.csv", R_s.astype(np.float32, copy=False), delimiter=",", fmt="%.7g")

            sel = np.argwhere(np.abs(R_s) >= pearson_threshold)
            if sel.size:
                sel = sel[sel[:, 0] < sel[:, 1]]
                wts = R_s[sel[:, 0], sel[:, 1]]
                order = np.argsort(-np.abs(wts))
                if pearson_topk is not None:
                    order = order[:pearson_topk]
                sel = sel[order]

            if write_graphml:
                G_p = nx.Graph()
                for i in present:
                    G_p.add_node(str(species[i]), relab=float(abund[i]))
                for i, j in sel:
                    ni, nj = str(species[int(i)]), str(species[int(j)])
                    if ni in present_set and nj in present_set:
                        G_p.add_edge(ni, nj, weight=float(R_s[int(i), int(j)]))
                nx.write_graphml(G_p, s_out / "graph_pearson_lioness.graphml")
                edges_kept_p = G_p.number_of_edges()
            else:
                if sel.size:
                    pmask = (present_mask[sel[:, 0]] & present_mask[sel[:, 1]])
                    edges_kept_p = int(pmask.sum())

        # Summaries
        upper_mask = np.zeros_like(W_boot_f, dtype=bool); upper_mask[iu, ju] = True
        pair_present = np.outer(present_mask, present_mask)
        edges_unfiltered = int((W_boot != 0.0).sum() // 2)
        edges_kept_boot = int(((W_boot_f != 0.0) & pair_present & upper_mask).sum())

        summary_rows[sname] = {
            "nodes_in": int(len(present_set)),
            "edges_unfiltered": edges_unfiltered,
            "edges_removed_by_pval": int(edges_unfiltered - edges_kept_boot),
            "prop_removed": float(0 if edges_unfiltered == 0 else (edges_unfiltered - edges_kept_boot) / edges_unfiltered),
            "nodes_out_boot": int(present_mask.sum()),
            "edges_out_boot": edges_kept_boot,
            "edges_out_pearson": edges_kept_p,
        }
        missing_cols = [c for c in species if c not in present_set]
        if missing_cols:
            filtered_otu.loc[sname, missing_cols] = 0.0

    if progress and hasattr(iterator, "close"):
        iterator.close()

    summary_df = pd.DataFrame.from_dict(summary_rows, orient='index')
    out_cond_summary.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_cond_summary / f"filtering_summary_{condition_group}.csv")
    filtered_otu.to_csv(out_cond_summary / f"feature_table_filtered_{condition_group}.csv")
    return filtered_otu, summary_df


def run_bootstrap_filtering_per_condition(
    feature_table: pd.DataFrame,
    metadata: pd.DataFrame,
    output_dir: Path,
    condition_id_col: str,
    num_bootstraps: int,
    p_value_threshold: float,
    min_samples_bootstrap: int,
    jobs: int = -1,
    seed: int = 42,
    save_boot_mats: bool = False,
    also_build_pearson: bool = True,
    pearson_threshold: float = 0.3,
    pearson_topk: Optional[int] = 10000,
    progress: bool = True,
    write_graphml: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    run_params_str = f"P_{p_value_threshold}_N_{num_bootstraps}"
    base_dir = output_dir / f"02_Bootstrap_Filtering_({run_params_str})"
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Output base: {base_dir}")
    print("[INFO] Condition sizes:", metadata[condition_id_col].astype(str).value_counts().to_dict())

    all_filtered, all_summ = {}, {}
    groups = list(metadata[condition_id_col].astype(str).unique())

    group_iter = groups
    if progress:
        group_iter = tqdm(groups, desc="Conditions", leave=False)

    for grp in group_iter:
        idx = metadata.index[metadata[condition_id_col].astype(str) == grp]
        valid_idx = feature_table.index.intersection(idx)
        if len(valid_idx) < min_samples_bootstrap:
            print(f"[WARN] Skipping condition '{grp}' (n={len(valid_idx)} < {min_samples_bootstrap})")
            continue
        df_cond = feature_table.loc[valid_idx]
        intermed_dir = base_dir / "Intermediates" / str(grp)

        boot_means, boot_stds, _ = bootstrap_condition_matrices(
            df_cond,
            num_bootstraps,
            intermed_dir,
            jobs=jobs,
            seed=seed,
            save_boot_mats=save_boot_mats,
            progress=progress,
        )

        pearson_maps = None
        if also_build_pearson:
            print(f"  Computing Pearson LIONESS per-sample matrices for condition '{grp}'...")
            pearson_maps = compute_feature_corr_lioness_fast(df_cond, jobs=jobs, progress=progress)

        filtered_otu, summary_df = filter_sample_graphs_with_bootstrap(
            df_cond,
            boot_means,
            boot_stds,
            base_dir,
            str(grp),
            p_thresh=p_value_threshold,
            pearson_edges_per_sample=pearson_maps,
            pearson_threshold=pearson_threshold,
            pearson_topk=pearson_topk,
            progress=progress,
            write_graphml=write_graphml,
        )
        all_filtered[str(grp)] = filtered_otu
        all_summ[str(grp)] = summary_df

    if progress and hasattr(group_iter, "close"):
        group_iter.close()

    return all_filtered, all_summ


# CLI


def main():
    p = argparse.ArgumentParser(description="Per-sample network builders (repo mode or file mode)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Bootstrap + Pearson graphs
    p_b = sub.add_parser("bootstrap", help="Build per-sample species/genera graphs via bootstrap filtering + Pearson LIONESS")
    # file mode
    p_b.add_argument("--otu", type=Path, help="Path to OTU/feature table (rows=samples, cols=features)")
    p_b.add_argument("--metadata", type=Path, help="Path to metadata file")
    p_b.add_argument("--out", type=Path, required=True)
    p_b.add_argument("--sample-id", type=str, default="Sample")
    p_b.add_argument("--condition-id", type=str, default="Study.Group")
    # params
    p_b.add_argument("--nboots", type=int, default=10)
    p_b.add_argument("--pval", type=float, default=0.05)
    p_b.add_argument("--min-samples", type=int, default=5)
    p_b.add_argument("--jobs", type=int, default=-1)
    p_b.add_argument("--pearson-threshold", type=float, default=0.3)
    p_b.add_argument("--pearson-topk", type=int, default=10000)
    p_b.add_argument("--seed", type=int, default=42, help="RNG seed for bootstrapping")
    p_b.add_argument("--save-boot-mats", action="store_true", help="Persist per-bootstrap matrices to disk (CSV)")
    p_b.add_argument("--no-pearson", action="store_true", help="Skip Pearson LIONESS graphs")
    p_b.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    p_b.add_argument("--matrices-only", action="store_true",
                     help="Do not build/write GraphML; only save per-sample adjacency matrices (CSV)")
    # repo mode
    p_b.add_argument("--repo-root", type=Path, help="Cloned curated repo root")
    p_b.add_argument("--dataset", type=str, help="Dataset folder under data/processed_data")

   
    try:
        # Python ≥3.9
        p_b.add_argument(
            "--prefer-species",
            dest="prefer_species",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Prefer species.tsv (default). Use --no-prefer-species to prefer genera.tsv.",
        )
    except AttributeError:
        # Python <3.9 fallback
        pref_group = p_b.add_mutually_exclusive_group()
        pref_group.add_argument("--prefer-species", dest="prefer_species", action="store_true", help="Prefer species.tsv (default).")
        pref_group.add_argument("--prefer-genera",  dest="prefer_species", action="store_false", help="Prefer genera.tsv.")
        p_b.set_defaults(prefer_species=True)

    args = p.parse_args()

    if args.cmd == "bootstrap":
        out_dir = args.out
        out_dir.mkdir(parents=True, exist_ok=True)

        show_progress = not args.no_progress

        # Load data (repo or file mode)
        if args.repo_root and args.dataset:
            feature_table, metadata = load_repo_feature_and_metadata(args.repo_root, args.dataset, prefer_species=args.prefer_species)
            # Save aligned for inspection in repo mode too
            processed_dir = out_dir / "01_Processed_Data"
            processed_dir.mkdir(parents=True, exist_ok=True)
            feature_table.to_csv(processed_dir / "feature_table_normalized.csv")
            metadata.to_csv(processed_dir / "metadata_aligned.csv")
            print(f"[INFO] Wrote aligned files to: {processed_dir}")

            cond = args.condition_id
            if cond not in metadata.columns:
                for cand in ["Study.Group","study_group","group","Group","case_control","Diagnosis","diagnosis","Status","status"]:
                    if cand in metadata.columns:
                        cond = cand; break
            if cond not in metadata.columns:
                raise SystemExit(f"Condition column '{args.condition_id}' not found. Available: {list(metadata.columns)}")
            args.condition_id = cond
        else:
            if args.otu is None or args.metadata is None:
                raise SystemExit("Provide --otu and --metadata, or use --repo-root and --dataset.")
            feature_table, metadata = process_data_generic(args.otu, args.metadata, out_dir,
                                                           sample_id=args.sample_id, condition_id=args.condition_id)

        # Orchestrate per-condition workflow
        _filtered, _summaries = run_bootstrap_filtering_per_condition(
            feature_table=feature_table,
            metadata=metadata,
            output_dir=out_dir,
            condition_id_col=args.condition_id,
            num_bootstraps=args.nboots,
            p_value_threshold=args.pval,
            min_samples_bootstrap=args.min_samples,
            jobs=args.jobs,
            seed=args.seed,
            save_boot_mats=args.save_boot_mats,
            also_build_pearson=not args.no_pearson,
            pearson_threshold=args.pearson_threshold,
            pearson_topk=(None if args.pearson_topk == 0 else args.pearson_topk),
            progress=show_progress,
            write_graphml=not args.matrices_only,
        )

        # Dataset-level avg non-zero features per class (epsilon > EPS_PRES)
        avg_nz = {}
        for cond, subidx in metadata.groupby(metadata[args.condition_id].astype(str)).groups.items():
            subdf = feature_table.loc[list(subidx)]
            nz_counts = (subdf > EPS_PRES).sum(axis=1)
            avg_nz[cond] = float(nz_counts.mean())
        avg_df = pd.DataFrame.from_dict(avg_nz, orient='index', columns=['avg_nonzero_features']).sort_index()
        avg_path = out_dir / "02_Bootstrap_Filtering_(P_{}_N_{})".format(args.pval, args.nboots) / "Condition_Summaries" / "avg_nonzero_features.csv"
        avg_path.parent.mkdir(parents=True, exist_ok=True)
        avg_df.to_csv(avg_path)
        print(f"[INFO] Wrote average nonzero feature counts per class to {avg_path}")


if __name__ == "__main__":
    main()
