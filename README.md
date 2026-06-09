# GP-Omics: Per-Sample Graph Construction 

These scripts build feature-feature co-occurrence networks for each sample in an omics dataset. Two graph construction methods are available: **LIONESS** (leave-one-out correlation) and **bootstrap-filtered weighted graphs** https://pubmed.ncbi.nlm.nih.gov/40811335/. 

---

## Installation

### 1. Create and activate the conda environment

```bash
conda create -n env_name python=3.10
conda activate env_name
```

### 2. Install Python dependencies

```bash
pip install numpy pandas scipy scikit-learn networkx joblib tqdm imbalanced-learn
```

> All scripts must be run with `env_name` active — it is the only environment with all required packages.

### 3. Clone the data submodule

```bash
git submodule update --init --recursive
```

This fetches the curated microbiome-metabolome datasets under `microbiome-metabolome-curated-data/data/processed_data/`.

---

## Graph Construction Methods

### Method 1: Spearman LIONESS (recommended for metabolomics)

LIONESS reconstructs a per-sample network by leave-one-out: it re-fits a population-level correlation network with each sample held out, then estimates that sample's individual network from the difference. The Spearman variant uses rank correlations, making it robust to non-normal distributions and outliers.

**When to use:** metabolomics data, or any dataset where features are non-normally distributed.

```bash
python graphs.py bootstrap \
  --otu path/to/features.tsv \
  --metadata path/to/metadata.tsv \
  --sample-id Sample \
  --condition-id Study.Group \
  --out path/to/output \
  --no-pearson \
  --spearman \
  --spearman-threshold 0.3 \
  --no-progress
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--spearman` | off | Enable Spearman LIONESS graphs |
| `--spearman-threshold` | 0.3 | Minimum \|ρ\| to keep an edge |
| `--spearman-topk` | 10000 | Max edges per sample (0 = unlimited) |
| `--no-pearson` | — | Skip Pearson LIONESS (pass this when using Spearman only) |

---

### Method 2: Pearson LIONESS

Same leave-one-out approach using Pearson correlation. Faster but assumes approximately normal feature distributions.

**When to use:** microbiome relative abundance data after normalisation, or when speed matters more than outlier robustness.

```bash
python graphs.py bootstrap \
  --otu path/to/features.tsv \
  --metadata path/to/metadata.tsv \
  --sample-id Sample \
  --condition-id Study.Group \
  --out path/to/output \
  --no-progress
```

(Pearson LIONESS is on by default; omit `--no-pearson` to include it.)

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--pearson-threshold` | 0.3 | Minimum \|r\| to keep an edge |
| `--pearson-topk` | 10000 | Max edges per sample |

---

### Method 3: Bootstrap-filtered weighted graph

Builds an abundance-weighted co-occurrence network and filters edges using bootstrap resampling: an edge is kept only if it is consistently present across bootstrap replicates (p < `--pval`). This is the default graph type and runs alongside whichever LIONESS method you choose.

**When to use:** when you want edge-level statistical confidence, or as an alternative to LIONESS.

This method always runs when you use `graphs.py bootstrap`. The LIONESS methods are opt-in additions.

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--nboots` | 10 | Number of bootstrap replicates |
| `--pval` | 0.05 | Edge filtering p-value threshold |
| `--seed` | 42 | RNG seed |

---

### Running all three methods together

```bash
python graphs.py bootstrap \
  --otu path/to/features.tsv \
  --metadata path/to/metadata.tsv \
  --sample-id Sample \
  --condition-id Study.Group \
  --out path/to/output \
  --spearman \
  --spearman-threshold 0.3 \
  --no-progress
```

(Pearson LIONESS + bootstrap are on by default; `--spearman` adds the Spearman variant.)

---

### Common optional flags

| Flag | Default | Description |
|---|---|---|
| `--jobs` | -1 (all cores) | Parallelism |
| `--matrices-only` | off | Write weight CSVs only, skip GraphML files |
| `--min-samples` | 5 | Minimum samples per condition |
| `--no-progress` | — | Suppress tqdm progress bars |

---

## Output structure

```
output/
├── 01_Processed_Data/
│   ├── feature_table_normalized.csv
│   └── metadata_aligned.csv
└── 02_Bootstrap_Filtering_(P_<pval>_N_<nboots>)/
    ├── Condition_Summaries/
    │   ├── filtering_summary_<condition>.csv
    │   ├── boot_summary_mean.csv
    │   └── boot_summary_std.csv
    └── Individual_Samples_by_Condition/<condition>/<sample>/
        ├── graph_bootstrap_filtered.graphml
        ├── weights_bootstrap_filtered.csv
        ├── graph_spearman_lioness.graphml      # if --spearman
        ├── weights_spearman_lioness.csv        # if --spearman
        ├── graph_pearson_lioness.graphml       # if Pearson enabled
        └── weights_pearson_lioness.csv         # if Pearson enabled
```

---

## Running on balanced data (class imbalance)

If your dataset is class-imbalanced, use `balance_with_graphs.py` to oversample the minority class before graph construction:

```bash
python balance_with_graphs.py \
  --feature-file path/to/features.tsv \
  --meta-file path/to/metadata.tsv \
  --id-col Sample \
  --label-col Study.Group \
  --positive "case" \
  --sampler SMOTE \
  --out-dir path/to/balanced_output \
  --run-graphs \
  --graphs-script graphs.py \
  --graphs-out path/to/graphs_output \
  --graphs-sample-id Sample \
  --graphs-condition-id Study.Group \
  --no-pearson --spearman --spearman-threshold 0.3 --no-progress
```

- `--positive` accepts comma-separated values for the positive class, e.g. `"Adenoma,Carcinoma"`.
- `--sampler` options: `SMOTE`, `AdaLoRAS`, or `NoSampling`.
- All flags after `--graphs-out` are passed directly to `graphs.py`.
- Without `--run-graphs`, the script outputs only the balanced CSV.

---

## Example: SINHA CRC dataset (Spearman LIONESS, unbalanced)

```bash
python graphs.py bootstrap \
  --otu microbiome-metabolome-curated-data/data/processed_data/SINHA_CRC_2016/mtb.tsv \
  --metadata microbiome-metabolome-curated-data/data/processed_data/SINHA_CRC_2016/metadata.tsv \
  --sample-id Sample --condition-id Study.Group \
  --out microbiome-metabolome-curated-data/data/processed_data/SINHA_CRC_2016/graphs_spearman \
  --no-pearson --spearman --no-progress
```

## Example: SINHA CRC dataset (SMOTE-balanced, Spearman LIONESS)

```bash
python balance_with_graphs.py \
  --feature-file microbiome-metabolome-curated-data/data/processed_data/SINHA_CRC_2016/mtb.tsv \
  --meta-file microbiome-metabolome-curated-data/data/processed_data/SINHA_CRC_2016/metadata.tsv \
  --id-col Sample --label-col Study.Group --positive "1" \
  --sampler SMOTE --variance-threshold 0 \
  --out-dir microbiome-metabolome-curated-data/data/processed_data/SINHA_CRC_2016/balanced_smote_mtb \
  --run-graphs --graphs-script graphs.py \
  --graphs-out microbiome-metabolome-curated-data/data/processed_data/SINHA_CRC_2016/graphs_smote_mtb \
  --graphs-sample-id Sample --graphs-condition-id Study.Group \
  --no-pearson --spearman --no-progress
```

---

## Datasets

Paired microbiome + metabolomics data is in the `microbiome-metabolome-curated-data` submodule (Borenstein Lab).

| Dataset | Disease | Samples |
|---|---|---|
| `KIM_ADENOMAS_2020` | Colorectal adenomas/carcinoma | 240 |
| `KOSTIC_INFANTS_DIABETES_2015` | Type 1 diabetes (infants) | 103 |
| `SINHA_CRC_2016` | Colorectal cancer | 131 |
| `FRANZOSA_IBD_2019` | IBD | — |

Feature files per dataset: `genera.tsv` (microbiome), `mtb.tsv` (metabolomics), `metadata.tsv`.

Positive class convention: disease/case is the positive class, set via `--positive`.
