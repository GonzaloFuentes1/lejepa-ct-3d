"""
Feature extraction for local LP evaluation.

Processes all splits (train/val/test) of a downstream dataset and saves
embeddings .h5 compatible with CT-NEXUS train_linear_probe.py and inference.py.

Output structure:
    output_dir/
        <dataset>/
            <task>/
                train/<caseid>.h5
                val/<caseid>.h5
                test/<caseid>.h5   (if present)

Usage:
    # All AMOS tasks (non-ROI + ROI)
    python scripts/eval_extract_features.py \\
        --ckpt checkpoints/lejepa_best.ckpt \\
        --downstream_root $LEJEPA_DATA_ROOT/cvpr26_downstream \\
        --output_dir $LEJEPA_DATA_ROOT/lejepa_embeddings

    # A single task
    python scripts/eval_extract_features.py \\
        --ckpt checkpoints/lejepa_best.ckpt \\
        --downstream_root $LEJEPA_DATA_ROOT/cvpr26_downstream \\
        --output_dir $LEJEPA_DATA_ROOT/lejepa_embeddings \\
        --tasks splenomegaly fatty_liver
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    SpatialPadd,
    Spacingd,
)
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Encoder configuration (must match the checkpoint)
# ---------------------------------------------------------------------------
PATCH_SIZE  = 16
SPACING     = (1.5, 1.5, 1.5)
HU_MIN      = -1000.0
HU_MAX      = 1000.0
VOLUME_SIZE = (128, 128, 128)  # 192mm FOV @ 1.5mm — must match pretraining

# ---------------------------------------------------------------------------
# Downstream tasks and whether they use ROI masking
# ---------------------------------------------------------------------------
TASKS_AMOS = {
    # task_name: uses_roi_mask
    "adrenal_hyperplasia":  True,
    "ascites":              False,
    "atherosclerosis":      False,
    "cholecystitis":        True,
    "colorectal_cancer":    False,
    "fatty_liver":          True,
    "gallstone":            True,
    "hydronephrosis":       True,
    "kidney_stone":         True,
    "liver_calcifications": True,
    "liver_cyst":           True,
    "liver_lesion":         True,
    "lymphadenopathy":      False,
    "renal_cyst":           True,
    "splenomegaly":         True,
}

TASKS_COVID  = {"covid":       False}
TASKS_LUNA25 = {"malignancy":  False}

DATASET_TASKS = {
    "AMOS-clf-tr-val": TASKS_AMOS,
    "COVID-CT":        TASKS_COVID,
    "LUNA25":          TASKS_LUNA25,
}


def _build_img_transforms() -> Compose:
    keys = ["image"]
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=list(SPACING), mode="bilinear"),
        ScaleIntensityRanged(
            keys=keys,
            a_min=HU_MIN, a_max=HU_MAX,
            b_min=0.0,   b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=keys, source_key="image"),
        SpatialPadd(keys=keys, spatial_size=VOLUME_SIZE),
        CenterSpatialCropd(keys=keys, roi_size=VOLUME_SIZE),
        EnsureTyped(keys=keys, dtype="float32"),
    ])


def _build_mask_transforms() -> Compose:
    keys = ["mask"]
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=list(SPACING), mode="nearest"),
        CropForegroundd(keys=keys, source_key="mask"),
        SpatialPadd(keys=keys, spatial_size=VOLUME_SIZE),
        CenterSpatialCropd(keys=keys, roi_size=VOLUME_SIZE),
        EnsureTyped(keys=keys, dtype="float32"),
    ])


@torch.no_grad()
def _encode(encoder, volume: torch.Tensor, mask: torch.Tensor | None, device) -> np.ndarray:
    """
    Encodes a (1,C,D,H,W) volume and returns an embedding (out_dim=2048,).
    encoder.forward() already applies proj_head per token → tokens are out_dim directly.
    project() only does mean pooling (with optional ROI masking).
    """
    volume = volume.to(device)
    tokens = encoder(volume)  # (1, N, out_dim=2048)

    roi_mask = None
    if mask is not None:
        mask = mask.to(device)
        m_down = F.avg_pool3d(mask.float(), kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
        roi_flat = (m_down.squeeze() > 0).reshape(-1)  # (N,)
        if roi_flat.any():
            roi_mask = roi_flat.unsqueeze(0)  # (1, N) bool

    emb = encoder.project(tokens, roi_mask=roi_mask).squeeze(0)  # (out_dim,)
    return emb.cpu().float().numpy()


def _load_encoder(ckpt_path: str, device: torch.device, fp16: bool = False):
    """Loads the ViT3D from a Lightning checkpoint."""
    repo_root = Path(__file__).parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from models.encoder.lejepa_module import LeJEPAModule

    module = LeJEPAModule.load_from_checkpoint(ckpt_path, map_location="cpu")
    encoder = module.encoder.eval().to(device)
    if fp16:
        encoder = encoder.half()
    del module
    torch.cuda.empty_cache()
    return encoder


def extract_dataset(
    encoder,
    dataset_dir: Path,
    task: str,
    uses_roi: bool,
    output_dir: Path,
    device: torch.device,
    fp16: bool,
    splits: list[str] | None = None,
) -> None:
    """Extracts features for all splits of a task."""
    img_dir   = dataset_dir / "images"
    mask_dir  = dataset_dir / "fg_masks" / task if uses_roi else None
    labels_df_path = dataset_dir / "labels" / f"{task}.csv"

    if not labels_df_path.exists():
        print(f"  [skip] No label CSV: {labels_df_path}")
        return

    import pandas as pd
    df = pd.read_csv(labels_df_path)

    # Detect available splits
    available_splits = df["split"].unique() if "split" in df.columns else ["train"]
    if splits:
        available_splits = [s for s in available_splits if s in splits]

    img_tfm  = _build_img_transforms()
    msk_tfm  = _build_mask_transforms() if uses_roi else None

    for split in available_splits:
        split_df = df[df["split"] == split] if "split" in df.columns else df
        out_split_dir = output_dir / task / split
        out_split_dir.mkdir(parents=True, exist_ok=True)

        pbar = tqdm(split_df.iterrows(), total=len(split_df),
                    desc=f"  {task}/{split}", leave=False)

        for _, row in pbar:
            case_id = row["case_id"]
            out_h5 = out_split_dir / case_id.replace(".nii.gz", ".h5")
            if out_h5.exists():
                continue  # skip if already extracted

            img_path = img_dir / case_id
            if not img_path.exists():
                pbar.write(f"    [warn] image not found: {img_path}")
                continue

            # Load image
            try:
                img_data = img_tfm({"image": str(img_path)})
                vol = img_data["image"].unsqueeze(0)  # (1,1,D,H,W)
                if fp16:
                    vol = vol.half()
            except Exception as e:
                pbar.write(f"    [error] {case_id}: {e}")
                continue

            # Load ROI mask (optional)
            mask = None
            if uses_roi and msk_tfm is not None:
                mask_path = mask_dir / case_id
                if mask_path.exists():
                    try:
                        msk_data = msk_tfm({"mask": str(mask_path)})
                        mask = msk_data["mask"].unsqueeze(0)  # (1,1,D,H,W)
                    except Exception as e:
                        pbar.write(f"    [warn] mask error {case_id}: {e}")

            # Extract embedding
            emb = _encode(encoder, vol, mask, device)

            # Guardar
            with h5py.File(out_h5, "w") as hf:
                hf.create_dataset("y_hat", data=emb, dtype="float32")

        pbar.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="LeJEPA LP feature extractor for local evaluation")
    ap.add_argument("--ckpt",             required=True,  help="Lightning checkpoint (.ckpt)")
    ap.add_argument("--downstream_root",  required=True,  help="Root dir of the downstream dataset")
    ap.add_argument("--output_dir",       required=True,  help="Where to save the .h5 embeddings")
    ap.add_argument("--tasks",            nargs="*",      help="List of tasks to extract (default: all)")
    ap.add_argument("--splits",           nargs="*",      default=None, help="Splits (default: all)")
    ap.add_argument("--device",           default="cuda")
    ap.add_argument("--fp16",             action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading encoder from {args.ckpt} ...")
    encoder = _load_encoder(args.ckpt, device, args.fp16)
    embed_dim = encoder.embed_dim
    print(f"Encoder loaded. embed_dim={embed_dim}")

    downstream_root = Path(args.downstream_root)
    output_dir      = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name, tasks_map in DATASET_TASKS.items():
        dataset_dir = downstream_root / dataset_name
        if not dataset_dir.exists():
            print(f"[skip] dataset not found: {dataset_dir}")
            continue

        print(f"\n=== {dataset_name} ===")
        dataset_out = output_dir / dataset_name

        for task, uses_roi in tasks_map.items():
            if args.tasks and task not in args.tasks:
                continue
            print(f"  Task: {task} (ROI={'yes' if uses_roi else 'no'})")
            extract_dataset(
                encoder=encoder,
                dataset_dir=dataset_dir,
                task=task,
                uses_roi=uses_roi,
                output_dir=dataset_out,
                device=device,
                fp16=args.fp16,
                splits=args.splits,
            )

    print("\nExtraction completed.")


if __name__ == "__main__":
    main()
