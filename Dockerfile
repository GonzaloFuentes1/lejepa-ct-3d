# Dockerfile — LeJEPA CT Foundation Model
# Task 1 (Linear Probing) — CVPR 2026 CT FM Competition
#
# Build:
#   docker build -t lejepa_lp .
#
# Tag for submission (name must end in _lp.tar.gz):
#   docker save lejepa_lp | gzip > lejepa_lp.tar.gz
#
# Test local:
#   mkdir -p test_outputs
#   docker container run --gpus "device=0" -m 32G --name lejepa_lp --rm \
#     -v $PWD/test_demo/:/workspace/inputs/ \
#     -v $PWD/test_outputs/:/workspace/outputs/ \
#     lejepa_lp:latest /bin/bash -c "sh extract_feat_LP.sh"

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Install uv for package management
RUN pip install --no-cache-dir uv

WORKDIR /workspace

# Copy the repo and the checkpoint
COPY . /workspace/lejepa-ct-3d/

# Install Python dependencies (without torch, already in the base image)
# Only those needed for inference (no wandb, hydra, etc.)
RUN pip install --no-cache-dir \
    pytorch-lightning>=2.3.0 \
    monai>=1.3.0 \
    nibabel>=5.2.0 \
    einops>=0.7.0 \
    scipy>=1.13.0 \
    numpy>=1.26.0 \
    h5py>=3.10.0 \
    omegaconf>=2.3.0

# The checkpoint is expected at /workspace/checkpoints/encoder.ckpt
# It can be mounted as a volume or copied here:
#   COPY checkpoints/best.ckpt /workspace/checkpoints/encoder.ckpt
# For the final submission, uncomment the line above and comment out the one below.
RUN mkdir -p /workspace/checkpoints

# Default environment variables
ENV PYTHONPATH="/workspace/lejepa-ct-3d:${PYTHONPATH}"

# Feature extraction script (invoked by the evaluator)
# CMD cannot be used because the evaluator calls it with:
#   /bin/bash -c "sh extract_feat_LP.sh"
# Therefore extract_feat_LP.sh must be reachable from /workspace/lejepa-ct-3d/
WORKDIR /workspace/lejepa-ct-3d
