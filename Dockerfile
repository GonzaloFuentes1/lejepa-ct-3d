# Dockerfile — LeJEPA CT Foundation Model
# Task 1 (Linear Probing) — CVPR 2026 CT FM Competition
#
# Build:
#   docker build -t lejepa_lp .
#
# Tag para submission (nombre debe terminar en _lp.tar.gz):
#   docker save lejepa_lp | gzip > lejepa_lp.tar.gz
#
# Test local:
#   mkdir -p test_outputs
#   docker container run --gpus "device=0" -m 32G --name lejepa_lp --rm \
#     -v $PWD/test_demo/:/workspace/inputs/ \
#     -v $PWD/test_outputs/:/workspace/outputs/ \
#     lejepa_lp:latest /bin/bash -c "sh extract_feat_LP.sh"

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Evitar prompts interactivos
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Instalar uv para gestión de paquetes
RUN pip install --no-cache-dir uv

WORKDIR /workspace

# Copiar el repo y el checkpoint
COPY . /workspace/lejepa-ct-3d/

# Instalar dependencias Python (sin torch — ya viene en la imagen base)
# Solo las necesarias para inferencia (no wandb, hydra, etc.)
RUN pip install --no-cache-dir \
    pytorch-lightning>=2.3.0 \
    monai>=1.3.0 \
    nibabel>=5.2.0 \
    einops>=0.7.0 \
    scipy>=1.13.0 \
    numpy>=1.26.0 \
    h5py>=3.10.0 \
    omegaconf>=2.3.0

# El checkpoint se espera en /workspace/checkpoints/encoder.ckpt
# Puede montarse como volumen o copiarse aquí:
#   COPY checkpoints/best.ckpt /workspace/checkpoints/encoder.ckpt
# Para la submission final, descomentar la línea de arriba y comentar la de abajo.
RUN mkdir -p /workspace/checkpoints

# Variables de entorno por defecto
ENV PYTHONPATH="/workspace/lejepa-ct-3d:${PYTHONPATH}"

# Script de extracción de features (lo invoca el evaluador)
# No se puede usar CMD porque el evaluador lo llama con:
#   /bin/bash -c "sh extract_feat_LP.sh"
# Por eso extract_feat_LP.sh debe estar accesible desde /workspace/lejepa-ct-3d/
WORKDIR /workspace/lejepa-ct-3d
