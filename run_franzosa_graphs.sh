#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/franzosa_graphs.log"
OUT_DIR="$SCRIPT_DIR/output/franzosa_graphs"

echo "======================================" | tee -a "$LOG_FILE"
echo "START: $(date)"                          | tee -a "$LOG_FILE"
echo "Installing dependencies..."              | tee -a "$LOG_FILE"
pip install numpy pandas networkx joblib scipy tqdm 2>&1 | tee -a "$LOG_FILE"
echo "Dependencies installed."                 | tee -a "$LOG_FILE"
echo "Output dir : $OUT_DIR"                   | tee -a "$LOG_FILE"
echo "Log file   : $LOG_FILE"                  | tee -a "$LOG_FILE"
echo "======================================" | tee -a "$LOG_FILE"

python "$SCRIPT_DIR/graphs.py" bootstrap \
    --repo-root "$SCRIPT_DIR/microbiome-metabolome-curated-data" \
    --dataset   FRANZOSA_IBD_2019 \
    --out       "$OUT_DIR" \
    --nboots    100 \
    --pval      0.05 \
    --min-samples 5 \
    --condition-id Study.Group \
    --prefer-species \
    --pearson-threshold 0.3 \
    --pearson-topk 10000 \
    --jobs -1 \
    2>&1 | tee -a "$LOG_FILE"

echo "======================================" | tee -a "$LOG_FILE"
echo "DONE: $(date)"                           | tee -a "$LOG_FILE"
echo "======================================" | tee -a "$LOG_FILE"
