#!/bin/bash
# extract_feat_LP.sh — Feature extraction script for Task 1 (Linear Probing)
#
# This script runs INSIDE the Docker container.
# The competition invokes it like this:
#   docker container run --gpus "device=0" -m 32G --name <team>_lp --rm \
#     [-e MASKS_DIR=/workspace/inputs/fg_masks/<disease>] \
#     -v $PWD/inputs/:/workspace/inputs/ \
#     -v $PWD/outputs/:/workspace/outputs/ \
#     <team>_lp:latest /bin/bash -c "sh extract_feat_LP.sh"
#
# Environment variables:
#   MASKS_DIR  (optional) — path to the ROI masks inside the container
#                           e.g. /workspace/inputs/fg_masks/splenomegaly

set -e

INPUT_DIR="/workspace/inputs"
OUTPUT_DIR="/workspace/outputs"
CKPT="/workspace/checkpoints/encoder.ckpt"
REPO_DIR="/workspace/lejepa-ct-3d"

# Build optional flags
EXTRA_FLAGS=""
if [ -n "${MASKS_DIR}" ]; then
    echo "ROI mode: using masks in ${MASKS_DIR}"
    EXTRA_FLAGS="--masks_dir ${MASKS_DIR}"
fi

echo "=== LeJEPA CT Feature Extractor ==="
echo "Input:   ${INPUT_DIR}"
echo "Output:  ${OUTPUT_DIR}"
echo "Ckpt:    ${CKPT}"
echo "Extra:   ${EXTRA_FLAGS}"
echo "==================================="

cd ${REPO_DIR}

python scripts/extract_features.py \
    --ckpt    "${CKPT}" \
    --input_dir  "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --fp16 \
    ${EXTRA_FLAGS}

echo "Extraction completed."
