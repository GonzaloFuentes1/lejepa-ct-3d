"""
Feature extraction para la competición CVPR 2026 — Task 1 (LP).

Carga un encoder ViT3D desde un checkpoint Lightning, procesa volúmenes CT
y guarda embeddings en formato .h5 compatible con run_LP.py del repo oficial.

Formato de salida (por .nii.gz de entrada):
    outputs/<casename>.h5
        y_hat: np.float32, shape (embed_dim,)  ← mean-pool de tokens válidos

ROI masking (enfermedades con ROI):
    Si se pasa --masks_dir, se carga la máscara binaria correspondiente al caso
    y se hace mean-pool solo sobre los tokens que solapan con la ROI.

Uso:
    # Sin ROI (atherosclerosis, colorectal_cancer, ascites, lymphadenopathy)
    python scripts/extract_features.py \\
        --ckpt checkpoints/best.ckpt \\
        --input_dir /workspace/inputs \\
        --output_dir /workspace/outputs

    # Con ROI (splenomegaly, adrenal_hyperplasia, etc.)
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
# Config (debe coincidir con el encoder del checkpoint)
# ---------------------------------------------------------------------------
PATCH_SIZE    = 16
SPACING       = (1.5, 1.5, 1.5)
HU_MIN        = -1000.0
HU_MAX        = 1000.0


def _load_env(path: str | None = None) -> None:
    """Lee .env manualmente (sin python-dotenv)."""
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
    """Transforms CT estándar sin crop ni augmentación."""
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
    """Transforms para máscara binaria (nn → no interpolation)."""
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
    Encoda un volumen y devuelve un embedding 1-D listo para LP.

    Usa encoder.project() → output dim = encoder.out_dim (2048 si proj_head activo).
    Si mask != None, aplica ROI-masked pooling antes de proyectar.
    """
    tokens = encoder(volume).squeeze(0)  # (N, embed_dim)

    # Construir máscara ROI a nivel de token si hay máscara de voxels
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
            "Concatenar features de los últimos 4 layers del encoder "
            "(capas 8, 16, 20, depth-1) → embedding de dim 4×embed_dim. "
            "Mejora downstream classification sin costo de pretraining."
        ),
    )
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- cargar encoder desde checkpoint ---
    print(f"Cargando checkpoint: {args.ckpt}")
    from models.encoder.lejepa_module import LeJEPAModule  # import relativo al repo

    module = LeJEPAModule.load_from_checkpoint(args.ckpt, map_location="cpu")
    encoder = module.encoder.to(device).eval()
    if args.fp16:
        encoder = encoder.half()
    del module
    torch.cuda.empty_cache()

    feat_dim = encoder.out_dim   # 2048 si proj_dim=2048, 1024 si no hay proj_head
    print(f"Encoder out_dim: {feat_dim} (competencia espera 2048)")

    # --- transforms ---
    img_tfm  = _build_transforms()
    msk_tfm  = _build_mask_transforms() if args.masks_dir else None

    # --- listar volúmenes ---
    input_dir = Path(args.input_dir)
    cases = sorted(input_dir.glob("*.nii.gz"))
    if not cases:
        cases = sorted(input_dir.glob("*.nii"))
    print(f"Procesando {len(cases)} volúmenes...")

    for i, case_path in enumerate(cases, 1):
        case_name = case_path.name.replace(".nii.gz", "").replace(".nii", "")
        out_path  = Path(args.output_dir) / f"{case_name}.h5"

        if out_path.exists():
            print(f"  [{i}/{len(cases)}] {case_name} — ya existe, saltando")
            continue

        print(f"  [{i}/{len(cases)}] {case_name}", end="", flush=True)

        try:
            # --- cargar imagen ---
            data = img_tfm({"image": str(case_path)})
            vol  = data["image"].unsqueeze(0).to(device)  # (1, 1, D, H, W)
            if args.fp16:
                vol = vol.half()

            # --- cargar máscara ROI (si aplica) ---
            mask_tensor = None
            if args.masks_dir is not None:
                mask_path = Path(args.masks_dir) / case_path.name
                if mask_path.exists():
                    mdata = msk_tfm({"mask": str(mask_path)})
                    mask_tensor = mdata["mask"].unsqueeze(0).to(device)  # (1, 1, D, H, W)
                    # Resize máscara al tamaño del volumen resampled si es necesario
                    if mask_tensor.shape[2:] != vol.shape[2:]:
                        mask_tensor = F.interpolate(
                            mask_tensor,
                            size=vol.shape[2:],
                            mode="nearest",
                        )
                else:
                    print(f" [WARN: máscara no encontrada, usando global pool]", end="")

            # --- extraer embedding ---
            feat = extract_one(encoder, vol, mask_tensor, PATCH_SIZE, device,
                               multi_layer=args.multi_layer)
            assert feat.shape == (feat_dim,), f"Shape inesperado: {feat.shape}"

            # --- guardar .h5 ---
            with h5py.File(out_path, "w") as hf:
                hf.create_dataset("y_hat", data=feat, dtype="float32")

            print(f" → {feat.shape} ✓")

        except Exception as e:
            print(f" ✗ ERROR: {e}")
            import traceback; traceback.print_exc()

    print(f"\nListo. Embeddings guardados en: {args.output_dir}")


if __name__ == "__main__":
    main()
