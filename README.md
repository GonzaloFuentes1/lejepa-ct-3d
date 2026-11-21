# lejepa-ct-3d

Self-supervised pretraining of 3D Vision Transformers on CT volumes with
LeJEPA (JEPA prediction plus SIGReg regularization), built for the
[CVPR 2026 Foundation Models for General CT Image Diagnosis challenge](https://www.codabench.org/competitions/12650)
(CT-FM). The code targets the AllData track (pretraining from scratch on
FLARE-Task4-CT-FM) and Task 1 (linear probing on 15 binary classification
tasks from AMOS).

This is research code. It was developed for the challenge and is shared as is.

## Status

- The pretraining pipeline, the four SSL modes, the feature extraction path and
  the Docker entrypoint for the Task 1 submission are implemented.
- A ViT-Large run was in progress at the last recorded point of the original
  development notes: around step 6K with a total loss of 3.28 and falling
  (taken from the previous README; no log file is versioned in this repo).
- **No final linear-probe results are reported.** The linear-probe callback is
  implemented but disabled in `scripts/pretrain.py` (commented out after a GPU
  desync issue). No baseline comparison numbers are included.
- Only two small test files exist (`tests/test_encoder.py`, `tests/test_sigreg.py`).
  Training has only been run on H100 GPUs.

## Architecture

- **Encoder**: 3D ViT with 3D patch embedding and 3D sinusoidal positional
  embeddings. It supports partial forward passes over a subset of tokens, so
  masked tokens are never processed. A projection head maps encoder tokens to a
  2048-dimensional feature for the challenge's linear-probe format.
  - ViT-Large configuration (`lejepa_diff_local`, `lejepa_diff_global`):
    embed_dim 1024, depth 24, 16 heads, patch size 16, about 307M parameters.
    Input crops are 96x160x160 voxels at 1.5 mm spacing, giving 6x10x10 = 600 tokens.
  - The committed `lejepa` and `jepa_ema` configs use a ViT-Base backbone
    (embed_dim 768, depth 12, 12 heads, patch size 8). Override
    `encoder.embed_dim`, `encoder.depth`, `encoder.num_heads` and
    `encoder.patch_size` to run them at ViT-Large size.
- **Predictor**: narrow cross-attention predictor (internal dim 384, roughly
  15M parameters). Mask tokens query the context tokens.
- **Masking**: 3D block masking with a fixed token budget per step (static
  shapes, so `torch.compile` does not recompile) and a curriculum from large to
  small target blocks.
- **Optimization**: AdamW with layer-wise learning-rate decay (0.75), bf16 mixed
  precision, DDP.

### SSL modes

| Mode | Config | Description |
|---|---|---|
| `lejepa` | `encoder=lejepa` | JEPA prediction loss plus SIGReg (Epps-Pulley test). No EMA, no stop-gradient. Default. |
| `jepa_ema` | `encoder=jepa_ema` | I-JEPA style target encoder updated by EMA with a cosine momentum schedule. |
| `lejepa_diff_local` | `encoder=lejepa_diff_local` | LeJEPA plus a flow-matching objective over target patch embeddings. |
| `lejepa_diff_global` | `encoder=lejepa_diff_global` | LeJEPA plus a flow-matching objective over the mean-pooled global embedding. |

## Data

Data is not included. Set `LEJEPA_DATA_ROOT` to the directory that holds it
(default `./data`); the Hydra configs and scripts read this variable.

- **Pretraining**: [FLARE-Task4-CT-FM](https://huggingface.co/datasets/FLARE-MedFM/FLARE-Task4-CT-FM).
  The original run used 108,200 pre-extracted float16 `.npy` crops of 96x160x160.
- **Linear-probe evaluation**: AMOS classification subset of the challenge
  downstream data (`kmin06/CVPR26-3DCTFMCompetition` on Hugging Face), 15 binary
  tasks, 1,463 volumes.
- **Segmentation visualization**: FLARE `abdomen_organ_seg` validation volumes
  (13 organ labels), used only for a PCA overlay diagnostic.

The Hugging Face datasets are gated and need an access token.

```bash
export LEJEPA_DATA_ROOT=/path/to/data
export HF_TOKEN=...                       # your own token, never commit it
pip install -e ".[download]"
python scripts/download_ct_datasets.py --list
python scripts/download_ct_datasets.py --downstream
python scripts/download_ct_datasets.py --pretrain --track coreset   # about 67 GB
python scripts/download_ct_datasets.py --pretrain --track alldata   # about 668 GB

# Convert .nii.gz to fast .npy crops (about 2 ms per load instead of seconds)
python scripts/preprocess_crops.py \
    --src $LEJEPA_DATA_ROOT/flare_ct_pretrain_full \
    --dst $LEJEPA_DATA_ROOT/flare_ct_pretrain_full_npy
```

## Installation

```bash
uv sync --python 3.12      # or: pip install -e ".[dev]"
```

`pyproject.toml` routes PyTorch to the CUDA 12.8 wheel index; edit
`[tool.uv.sources]` for another CUDA version or CPU. Optional logging to Weights
and Biases reads `WANDB_API_KEY` (and optionally `WANDB_ENTITY`) from the
environment or a local `.env` file (git-ignored); set `wandb.project=null` to
disable it.

## Training

```bash
# default: lejepa on the AllData crops
python scripts/pretrain.py encoder=lejepa data.batch_size_per_gpu=80

python scripts/pretrain.py encoder=jepa_ema data.batch_size_per_gpu=80
python scripts/pretrain.py encoder=lejepa_diff_local data.batch_size_per_gpu=80
python scripts/pretrain.py encoder=lejepa_diff_global data.batch_size_per_gpu=80

# smoke test and resume
python scripts/pretrain.py trainer=smoke
python scripts/pretrain.py resume=checkpoints/last.ckpt
```

On a Slurm cluster, wrap the command with `srun --gpus=1 ...`. Trainer presets
are in `configs/trainer/` (`h100_ddp`, `h100_ddp_ema`, `smoke`).

## Evaluation

```bash
# Extract features and run a local linear probe on AMOS
python scripts/run_local_lp.py \
    --ckpt checkpoints/best.ckpt \
    --data_dir $LEJEPA_DATA_ROOT/cvpr26_downstream/AMOS-clf-tr-val \
    --feat_dir $LEJEPA_DATA_ROOT/cvpr26_lp_feats

# Or the staged pipeline
bash scripts/eval_pipeline.sh checkpoints/best.ckpt
```

`scripts/compare_baselines.py` builds a comparison table from linear-probe
result folders you provide.

### Challenge submission image

```bash
cp checkpoints/best.ckpt checkpoints/encoder.ckpt
docker build -t lejepa_lp .
docker save lejepa_lp | gzip > lejepa_lp.tar.gz
```

`extract_feat_LP.sh` is the entrypoint called by the evaluator.

## Tests

```bash
pip install -e ".[dev]"
python -m pytest -q tests
```

## Repository layout

```
configs/    Hydra configs (encoder modes, data tracks, trainer presets)
data/       npy crop dataset and MONAI transforms
models/     ViT3D encoder, predictor, flow-matching nets, SIGReg, VCReg, EMA
utils/      masking, PCA visualization and linear-probe callbacks
scripts/    pretraining, preprocessing, download, feature extraction, linear probe
examples/   CT volume inspection helper
docs/       architecture diagram (Excalidraw)
```

## References

- R. Balestriero and Y. LeCun, "LeJEPA", [arXiv:2511.08544](https://arxiv.org/abs/2511.08544).
- M. Assran et al., "I-JEPA", [arXiv:2301.08243](https://arxiv.org/abs/2301.08243).
- Y. Lipman et al., "Flow Matching for Generative Modeling", [arXiv:2210.02747](https://arxiv.org/abs/2210.02747).
- CVPR 2026 Foundation Models for General CT Image Diagnosis challenge, [Codabench 12650](https://www.codabench.org/competitions/12650).

## License

Apache License 2.0. See `LICENSE`.
