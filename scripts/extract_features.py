"""
Feature extraction for the CVPR 2026 competition, Task 1 (LP).

Loads a ViT3D encoder from a Lightning checkpoint, processes CT volumes
and saves embeddings in .h5 format compatible with run_LP.py of the official repo.

Output format (per input .nii.gz):
    outputs/<casename>.h5
        y_hat: np.float32, shape (embed_dim,)  ← mean-pool of valid tokens

ROI masking (diseases with ROI):
    If --masks_dir is given, the binary mask corresponding to the case is loaded
    and mean-pooling is done only over the tokens overlapping the ROI.

Usage:
    # Without ROI (atherosclerosis, colorectal_cancer, ascites, lymphadenopathy)
    python scripts/extract_features.py \\
        --ckpt checkpoints/best.ckpt \\
        --input_dir /workspace/inputs \\
        --output_dir /workspace/outputs

    # With ROI (splenomegaly, adrenal_hyperplasia, etc.)
    python scripts/extract_features.py \\
        --ckpt checkpoints/best.ckpt \\
        --input_dir /workspace/inputs \\
        --output_dir /workspace/outputs \\
        --masks_dir /workspace/inputs/fg_masks
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)

# ---------------------------------------------------------------------------
# Config (must match the checkpoint encoder)
# ---------------------------------------------------------------------------
PATCH_SIZE    = 16
SPACING       = (1.5, 1.5, 1.5)
HU_MIN        = -1000.0
HU_MAX        = 1000.0


def _load_env(path: str | None = None) -> None:
    """Reads .env manually (without python-dotenv)."""
    search = [Path(path)] if path else [
        Path(__file__).parent,
        Path(__file__).parent.parent,
    ]
    for base in search:
        env_file = base / ".env" if base.is_dir() else base
        if env_file.is_file():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, _, v = line.partition("=")
                        os.environ.setdefault(k.strip(), v.strip().strip('"\''))
            break


def _build_transforms() -> Compose:
    """Standard CT transforms without crop or augmentation."""
    keys = ["image"]
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=list(SPACING), mode="bilinear"),
        ScaleIntensityRanged(
            keys=keys,
            a_min=HU_MIN, a_max=HU_MAX,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        EnsureTyped(keys=keys, dtype="float32"),
    ])


def _build_mask_transforms() -> Compose:
    """Transforms for binary mask (nn → no interpolation)."""
    keys = ["mask"]
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=list(SPACING), mode="nearest"),
        EnsureTyped(keys=keys, dtype="float32"),
    ])


@torch.no_grad()
def extract_one(
    encoder: torch.nn.Module,
    volume: torch.Tensor,        # (1, C, D, H, W) float32 on device
    mask: torch.Tensor | None,   # (1, 1, D, H, W) binary on device, or None
    patch_size: int,
    device: torch.device,
    multi_layer: bool = False,   # legacy: kept for backward compatibility
) -> np.ndarray:
    """
    Encodes a volume and returns a 1-D embedding ready for LP.

    Uses encoder.project() → output dim = encoder.out_dim (2048 if proj_head is active).
    If mask != None, applies ROI-masked pooling before projecting.
    """
    tokens = encoder(volume).squeeze(0)  # (N, embed_dim)

    # Build the ROI mask at token level if there is a voxel mask
    roi_mask = None
    if mask is not None:
        m_down = F.avg_pool3d(mask.float(), kernel_size=patch_size, stride=patch_size)
        roi_flat = (m_down.squeeze() > 0).reshape(-1)   # (N,) bool
        roi_mask = roi_flat.unsqueeze(0) if roi_flat.any() else None  # (1, N)

    # project(): pool + proj_head → (1, out_dim)
    feat = encoder.project(tokens.unsqueeze(0), roi_mask=roi_mask)
    return feat.squeeze(0).cpu().float().numpy()  # (out_dim,) float32


def main() -> None:
    _load_env()

    ap = argparse.ArgumentParser(description="LeJEPA CT feature extractor")
    ap.add_argument("--ckpt",        required=True,  help="Path to Lightning checkpoint (.ckpt)")
    ap.add_argument("--input_dir",   required=True,  help="Dir with .nii.gz input volumes")
    ap.add_argument("--output_dir",  required=True,  help="Dir to write .h5 embeddings")
    ap.add_argument("--masks_dir",   default=None,   help="Dir with .nii.gz ROI masks (ROI diseases only)")
    ap.add_argument("--device",      default="cuda", help="cuda | cpu")
    ap.add_argument("--fp16",        action="store_true", help="Run encoder in fp16")
    ap.add_argument(
        "--multi_layer",
        action="store_true",
        help=(
            "Concatenate features from the last 4 encoder layers "
            "(layers 8, 16, 20, depth-1) → embedding of dim 4×embed_dim. "
            "Improves downstream classification at no pretraining cost."
        ),
    )
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- load encoder from checkpoint ---
    print(f"Loading checkpoint: {args.ckpt}")
    from models.encoder.lejepa_module import LeJEPAModule  # import relative to the repo

    module = LeJEPAModule.load_from_checkpoint(args.ckpt, map_location="cpu")
    encoder = module.encoder.to(device).eval()
    if args.fp16:
        encoder = encoder.half()
    del module
    torch.cuda.empty_cache()

    feat_dim = encoder.out_dim   # 2048 if proj_dim=2048, 1024 if there is no proj_head
    print(f"Encoder out_dim: {feat_dim} (competition expects 2048)")

    # --- transforms ---
    img_tfm  = _build_transforms()
    msk_tfm  = _build_mask_transforms() if args.masks_dir else None

    # --- list volumes ---
    input_dir = Path(args.input_dir)
    cases = sorted(input_dir.glob("*.nii.gz"))
    if not cases:
        cases = sorted(input_dir.glob("*.nii"))
    print(f"Processing {len(cases)} volumes...")

    for i, case_path in enumerate(cases, 1):
        case_name = case_path.name.replace(".nii.gz", "").replace(".nii", "")
        out_path  = Path(args.output_dir) / f"{case_name}.h5"

        if out_path.exists():
            print(f"  [{i}/{len(cases)}] {case_name} - already exists, skipping")
            continue

        print(f"  [{i}/{len(cases)}] {case_name}", end="", flush=True)

        try:
            # --- load image ---
            data = img_tfm({"image": str(case_path)})
            vol  = data["image"].unsqueeze(0).to(device)  # (1, 1, D, H, W)
            if args.fp16:
                vol = vol.half()

            # --- load ROI mask (if applicable) ---
            mask_tensor = None
            if args.masks_dir is not None:
                mask_path = Path(args.masks_dir) / case_path.name
                if mask_path.exists():
                    mdata = msk_tfm({"mask": str(mask_path)})
                    mask_tensor = mdata["mask"].unsqueeze(0).to(device)  # (1, 1, D, H, W)
                    # Resize the mask to the resampled volume size if necessary
                    if mask_tensor.shape[2:] != vol.shape[2:]:
                        mask_tensor = F.interpolate(
                            mask_tensor,
                            size=vol.shape[2:],
                            mode="nearest",
                        )
                else:
                    print(f" [WARN: mask not found, using global pool]", end="")

            # --- extract embedding ---
            feat = extract_one(encoder, vol, mask_tensor, PATCH_SIZE, device,
                               multi_layer=args.multi_layer)
            assert feat.shape == (feat_dim,), f"Unexpected shape: {feat.shape}"

            # --- save .h5 ---
            with h5py.File(out_path, "w") as hf:
                hf.create_dataset("y_hat", data=feat, dtype="float32")

            print(f" → {feat.shape} ✓")

        except Exception as e:
            print(f" ✗ ERROR: {e}")
            import traceback; traceback.print_exc()

    print(f"\nDone. Embeddings saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
