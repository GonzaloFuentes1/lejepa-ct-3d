#!/usr/bin/env bash
# eval_pipeline.sh — Full LP evaluation pipeline for LeJEPA
#
# Usage:
#   bash scripts/eval_pipeline.sh <checkpoint.ckpt> [output_dir]
#
# Example:
#   bash scripts/eval_pipeline.sh checkpoints/lejepa_best.ckpt $LEJEPA_DATA_ROOT/lejepa_eval
#
# Requires: source .venv/bin/activate

set -euo pipefail

CKPT="${1:-}"
DATA_ROOT="${LEJEPA_DATA_ROOT:-./data}"
EVAL_ROOT="${2:-$DATA_ROOT/lejepa_eval}"
DOWNSTREAM_ROOT="$DATA_ROOT/cvpr26_downstream"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ -z "$CKPT" ]]; then
    echo "Usage: bash scripts/eval_pipeline.sh <checkpoint.ckpt> [output_dir]"
    exit 1
fi

if [[ ! -f "$CKPT" ]]; then
    echo "Error: checkpoint not found: $CKPT"
    exit 1
fi

EMBEDS_ROOT="${EVAL_ROOT}/embeddings"
LP_RESULTS="${EVAL_ROOT}/lp_results"

echo "=============================================="
echo "  LeJEPA LP Evaluation Pipeline"
echo "  Checkpoint : $CKPT"
echo "  Downstream : $DOWNSTREAM_ROOT"
echo "  Embeddings : $EMBEDS_ROOT"
echo "  LP results : $LP_RESULTS"
echo "=============================================="

cd "$REPO_ROOT"

# -------------------------------------------------------
# STEP 1: Extract features for all datasets
# -------------------------------------------------------
echo ""
echo "[1/3] Extracting features..."

for DATASET in AMOS-clf-tr-val COVID-CT LUNA25; do
    DATASET_DIR="${DOWNSTREAM_ROOT}/${DATASET}"
    if [[ ! -d "$DATASET_DIR" ]]; then
        echo "  [skip] Dataset not found: $DATASET_DIR"
        continue
    fi
    echo "  Dataset: $DATASET"
    python scripts/eval_extract_features.py \
        --ckpt          "$CKPT" \
        --downstream_root "$DOWNSTREAM_ROOT" \
        --output_dir    "$EMBEDS_ROOT" \
        --device        cuda
done

# -------------------------------------------------------
# STEP 2: Train and evaluate linear heads
# -------------------------------------------------------
echo ""
echo "[2/3] Training linear probes..."

for DATASET in AMOS-clf-tr-val COVID-CT LUNA25; do
    EMBEDS_DATASET="${EMBEDS_ROOT}/${DATASET}"
    LABELS_DIR="${DOWNSTREAM_ROOT}/${DATASET}/labels"
    if [[ ! -d "$EMBEDS_DATASET" ]]; then
        echo "  [skip] Embeddings not found: $EMBEDS_DATASET"
        continue
    fi
    echo "  Dataset: $DATASET"
    python scripts/train_linear_probe.py \
        --embeds_root "$EMBEDS_DATASET" \
        --labels_root "$LABELS_DIR" \
        --output_dir  "${LP_RESULTS}/${DATASET}" \
        --dataset     "$DATASET" \
        --device      cuda \
        --max_epochs  100
done

# -------------------------------------------------------
# STEP 3: Compare against baselines
# -------------------------------------------------------
echo ""
echo "[3/3] Generating comparison table..."

python scripts/compare_baselines.py \
    --lp_results_root "$LP_RESULTS" \
    --output          "${EVAL_ROOT}/comparison_table.csv"

echo ""
echo "=============================================="
echo "  Pipeline complete."
echo "  Table: ${EVAL_ROOT}/comparison_table.csv"
echo "=============================================="
