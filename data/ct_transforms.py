"""
Transforms MONAI para CT 3D (FLARE-Task4-CT-FM y CVPR26 downstream).

Normalización CT:
    - HU windowing: clip a [hu_min, hu_max] luego rescale a [0, 1]
    - Por defecto: [-1000, 1000] HU (cubre aire, grasa, tejidos, hueso)
    - Multi-windowing (Pillar-0): 4 canales con ventanas especializadas
      soft tissue [-150,250], lung [-1350,150], bone [-500,1300], mediastinum [-125,225]
    - Resampling a spacing isotrópico (1.5 mm por defecto)
    - Orientación canónica RAS

Diferencias con MRI (mri_transforms.py):
    - 1 canal (no multichannel como BraTS)
    - ScaleIntensityRanged en lugar de NormalizeIntensityd (HU tiene escala absoluta)
    - Spacing más grueso (1.5 mm) para CT de cuerpo completo
    - Patch size 96³ (volúmenes CT más grandes, memoria limitada)
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import numpy as np
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambda,
    LoadImaged,
    MapTransform,
    Orientationd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    RandZoomd,
    CenterSpatialCropd,
    ScaleIntensityRanged,
    SpatialCrop,
    SpatialPad,
    SpatialPadd,
    Spacingd,
)


# Ventanas HU para multi-windowing (Pillar-0, Table 2)
# Cada tupla: (hu_min, hu_max)
PILLAR0_WINDOWS = [
    (-150,   250),   # soft tissue
    (-1350,  150),   # lung
    (-500,  1300),   # bone
    (-125,   225),   # mediastinum
]


class LoadNpyd(MapTransform):
    """
    Carga un volumen pre-procesado desde .npy (int16 HU raw) → tensor float32 4 canales.

    Aplica multi-windowing Pillar-0 en runtime: convierte HU raw a 4 canales
    normalizados [0,1] con ventanas especializadas (soft tissue, lung, bone, mediastinum).

    Input:  .npy int16 (D, H, W) HU raw
    Output: tensor float16 (4, D, H, W) normalizado [0,1] por canal
    """

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.keys:
            arr = np.load(str(d[key])).astype(np.float32)   # (D, H, W) HU raw
            t   = torch.from_numpy(arr)                      # (D, H, W)
            channels = []
            for hu_min, hu_max in PILLAR0_WINDOWS:
                c = t.clamp(hu_min, hu_max)
                c = (c - hu_min) / (hu_max - hu_min)         # → [0, 1]
                channels.append(c.to(torch.float16))
            d[key] = torch.stack(channels, dim=0)            # (4, D, H, W) float16
        return d


class MultiWindowd(MapTransform):
    """
    Aplica multi-windowing Pillar-0 sobre un tensor HU raw ya cargado.

    Para el pipeline slow/cropfirst: convierte (1, D, H, W) HU raw
    a (4, D, H, W) float32 multi-ventana.
    """

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.keys:
            t = d[key]  # (1, D, H, W)
            if t.shape[0] == 1:
                t = t.squeeze(0)  # (D, H, W)
            channels = []
            for hu_min, hu_max in PILLAR0_WINDOWS:
                c = t.float().clamp(hu_min, hu_max)
                c = (c - hu_min) / (hu_max - hu_min)
                channels.append(c)
            d[key] = torch.stack(channels, dim=0)  # (4, D, H, W)
        return d


class FastAugd(MapTransform):
    """
    Augmentaciones espaciales rápidas en PyTorch puro (sin MetaTensor overhead).

    Para el pipeline rápido .npy: ~3ms vs ~20ms con MONAI RandFlipd/Rotate90d.
    Opera sobre tensores (1, D, H, W) float16 o float32.
    """

    def __init__(self, keys, flip_prob: float = 0.5, rot90_prob: float = 0.3):
        super().__init__(keys)
        self.flip_prob  = flip_prob
        self.rot90_prob = rot90_prob

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.keys:
            x = d[key]  # (1, D, H, W)
            # Random flips sobre los 3 ejes espaciales
            for axis in (1, 2, 3):
                if torch.rand(1).item() < self.flip_prob:
                    x = x.flip(axis)
            # Random 90° rotations solo en plano axial in-plane (H-W, dims 2-3).
            # No rotar en planos que involucran D (dims 1-2 o 1-3): con crops no
            # cúbicos (D=96 ≠ H=W=160) rot90 intercambiaría D↔H o D↔W y produciría
            # tensores de distinto shape que no se pueden apilar en el batch.
            if torch.rand(1).item() < self.rot90_prob:
                k = int(torch.randint(1, 4, (1,)).item())
                x = x.rot90(k, dims=(2, 3))
            d[key] = x
        return d


class RandNativePatchd(MapTransform):
    """
    Crop in native voxel space before Spacingd, dramatically reducing resample cost.

    Strategy: compute how many native voxels cover `target_mm` in each axis,
    take a random crop of that size, then let Spacingd resample only that small
    patch instead of the full volume.

    Example (0.73mm in-plane, 5mm slice):
      crop_size = ceil([150/0.73, 150/0.73, 150/5]) = [206, 206, 30]
      voxels resampled: 206×206×30 ≈ 1.3M   (vs full 512×512×60 ≈ 15M)
      Spacingd speedup: ~10×  (590ms → ~60ms)

    After this transform + Spacingd(1.5mm): output ≈ 100³.
    Use SpatialPadd(96³) + CenterSpatialCropd(96³) to get exact size.
    """

    def __init__(
        self,
        keys,
        target_mm: Tuple[float, float, float] = (150.0, 150.0, 150.0),
        center: bool = False,
    ):
        super().__init__(keys)
        self.target_mm = target_mm
        self.center = center

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.keys:
            img = d[key]  # (C, D0, D1, D2) MetaTensor

            affine = getattr(img, "meta", {}).get("affine", None)
            if affine is None:
                continue  # no metadata — skip crop, Spacingd will still work

            aff_np = affine.numpy() if hasattr(affine, "numpy") else np.array(affine)
            spacing = np.linalg.norm(aff_np[:3, :3], axis=0).clip(0.1, 15.0)  # (3,) mm/voxel

            shape = list(img.shape[1:])  # (D0, D1, D2)

            # Native voxels needed to cover target_mm
            crop_size = [math.ceil(self.target_mm[i] / float(spacing[i])) for i in range(3)]
            # Clamp to actual volume size (with min 1)
            crop_size = [max(1, min(crop_size[i], shape[i])) for i in range(3)]

            if self.center:
                starts = [max(0, (shape[i] - crop_size[i]) // 2) for i in range(3)]
            else:
                starts = [
                    int(torch.randint(0, max(shape[i] - crop_size[i] + 1, 1), (1,)).item())
                    for i in range(3)
                ]
            ends = [min(starts[i] + crop_size[i], shape[i]) for i in range(3)]

            # SpatialCrop handles MetaTensor affine update correctly
            cropped = SpatialCrop(roi_start=starts, roi_end=ends)(img)

            # Pad to crop_size if volume was smaller than requested
            needs_pad = any(cropped.shape[1 + i] < crop_size[i] for i in range(3))
            if needs_pad:
                cropped = SpatialPad(spatial_size=crop_size, method="end")(cropped)

            d[key] = cropped
        return d


# ---------------------------------------------------------------------------
# Transforms base CT
# ---------------------------------------------------------------------------

def _ct_base_transforms(
    img_keys: List[str],
    spacing: Tuple[float, float, float],
    hu_min: float,
    hu_max: float,
    extra_keys: List[str] = [],
) -> List:
    """
    Pipeline base para cualquier volumen CT:
      1. Cargar NIfTI
      2. Orientación RAS canónica
      3. Resampling al spacing objetivo
      4. HU windowing → [0, 1]
    """
    all_keys = img_keys + extra_keys
    n_extra  = len(extra_keys)

    return [
        LoadImaged(keys=all_keys),
        EnsureChannelFirstd(keys=all_keys),
        Orientationd(keys=all_keys, axcodes="RAS"),
        Spacingd(
            keys=all_keys,
            pixdim=list(spacing),
            mode=["bilinear"] * len(img_keys) + ["nearest"] * n_extra,
        ),
        # HU windowing: clip + rescale a [0, 1]
        ScaleIntensityRanged(
            keys=img_keys,
            a_min=hu_min,
            a_max=hu_max,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
    ]


# ---------------------------------------------------------------------------
# Augmentaciones estándar
# ---------------------------------------------------------------------------

def _spatial_aug_ct(keys: List[str], label_keys: List[str] = []) -> List:
    """Augmentaciones espaciales para CT: flips + rotaciones 90° + zoom."""
    return [
        RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
        # Rotaciones 90°: gratuitas (permutación de ejes, sin interpolación)
        # Fuerzan invarianza a orientación — CT de distintos centros varían
        RandRotate90d(keys=keys, prob=0.3, spatial_axes=(0, 1)),
        RandRotate90d(keys=keys, prob=0.2, spatial_axes=(0, 2)),
        # Zoom conservador con keep_size=True para mantener 96³ fijo
        # (sin esto, el output size varía y el ViT3D grid 6×6×6 falla)
        RandZoomd(keys=keys, min_zoom=0.85, max_zoom=1.15, prob=0.3, keep_size=True),
    ]


def _intensity_aug_ct(img_keys: List[str]) -> List:
    """Augmentaciones de intensidad ligeras para CT (rango HU ya fijo)."""
    return [
        RandScaleIntensityd(keys=img_keys, factors=0.1, prob=0.3),
        RandShiftIntensityd(keys=img_keys, offsets=0.1, prob=0.3),
        RandGaussianNoised(keys=img_keys,  std=0.01,    prob=0.2),
        RandGaussianSmoothd(
            keys=img_keys,
            sigma_x=(0.5, 1.0),
            sigma_y=(0.5, 1.0),
            sigma_z=(0.5, 1.0),
            prob=0.2,
        ),
    ]


# ---------------------------------------------------------------------------
# Pipelines por modo
# ---------------------------------------------------------------------------

def _ct_pretrain_transforms(
    patch_size: Tuple[int, int, int],
    spacing: Tuple[float, float, float],
    hu_min: float,
    hu_max: float,
) -> Compose:
    """SSL pretrain: imagen sola, crop aleatorio, augmentaciones."""
    img_key = ["image"]
    base    = _ct_base_transforms(img_key, spacing, hu_min, hu_max)

    return Compose([
        *base,
        CropForegroundd(keys=img_key, source_key="image"),
        SpatialPadd(keys=img_key, spatial_size=patch_size),
        RandSpatialCropd(keys=img_key, roi_size=patch_size, random_size=False),
        *_spatial_aug_ct(img_key),
        *_intensity_aug_ct(img_key),
        EnsureTyped(keys=img_key, dtype="float32"),
    ])


def _ct_classify_transforms(
    patch_size: Tuple[int, int, int],
    spacing: Tuple[float, float, float],
    hu_min: float,
    hu_max: float,
) -> Compose:
    """
    Clasificación (linear probe / fine-tune): imagen completa sin crop,
    ya que necesitamos representación global del volumen.

    El encoder extrae features del volumen completo → pool global → lineal.
    """
    img_key = ["image"]
    base    = _ct_base_transforms(img_key, spacing, hu_min, hu_max)

    return Compose([
        *base,
        CropForegroundd(keys=img_key, source_key="image"),
        SpatialPadd(keys=img_key, spatial_size=patch_size),
        *_spatial_aug_ct(img_key),
        *_intensity_aug_ct(img_key),
        EnsureTyped(keys=img_key, dtype="float32"),
    ])


def _ct_val_transforms(
    patch_size: Tuple[int, int, int],
    spacing: Tuple[float, float, float],
    hu_min: float,
    hu_max: float,
) -> Compose:
    """Validación/test: sin augmentaciones, crop central determinista."""
    img_key = ["image"]
    base    = _ct_base_transforms(img_key, spacing, hu_min, hu_max)

    return Compose([
        *base,
        CropForegroundd(keys=img_key, source_key="image"),
        SpatialPadd(keys=img_key, spatial_size=patch_size),
        CenterSpatialCropd(keys=img_key, roi_size=patch_size),
        EnsureTyped(keys=img_key, dtype="float32"),
    ])


# ---------------------------------------------------------------------------
# Pipelines rápidos (para .npy pre-procesados: sin Load/Orient/Resample)
# ---------------------------------------------------------------------------

def _ct_pretrain_fast(patch_size: Tuple[int, int, int]) -> Compose:
    """
    Pipeline rápido para .npy pre-procesados (int16 HU raw).

    LoadNpyd aplica multi-windowing Pillar-0 en runtime: int16 HU raw
    → 4 canales float16 [0,1] (soft tissue, lung, bone, mediastinum).

    Usa FastAugd (PyTorch puro) en vez de MONAI para las augmentaciones
    espaciales: evita overhead MetaTensor (~20ms → ~3ms).

    Speedup vs pipeline original: ~900ms → ~35ms por sample.
    """
    img_key = ["image"]
    return Compose([
        LoadNpyd(keys=img_key),                                         # int16 HU raw → (4,D,H,W) float16
        SpatialPadd(keys=img_key, spatial_size=patch_size),             # pad si vol < patch_size
        RandSpatialCropd(keys=img_key, roi_size=patch_size, random_size=False),
        FastAugd(keys=img_key, flip_prob=0.5, rot90_prob=0.3),         # flips + rot90 rápidos
        *_intensity_aug_ct(img_key),                                    # MONAI intensity augs
        EnsureTyped(keys=img_key, dtype="float32"),                     # float16 → float32
    ])


def _ct_val_fast(patch_size: Tuple[int, int, int]) -> Compose:
    """Pipeline rápido de validación para .npy (int16 HU raw → 4 canales)."""
    img_key = ["image"]
    return Compose([
        LoadNpyd(keys=img_key),
        SpatialPadd(keys=img_key, spatial_size=patch_size),
        CenterSpatialCropd(keys=img_key, roi_size=patch_size),
        EnsureTyped(keys=img_key, dtype="float32"),
    ])


def _ct_pretrain_cropfirst(
    patch_size: Tuple[int, int, int],
    spacing: Tuple[float, float, float],
    hu_min: float,
    hu_max: float,
) -> Compose:
    """
    Pipeline Option-A: crop native patch → resample only patch (no .npy needed).

    Key ordering: ScaleIntensityRanged is placed AFTER RandNativePatchd so it
    runs on the small patch (~7M voxels) instead of the full volume (~200M voxels).
    CropForegroundd uses raw HU (select_fn > -500: keeps all tissue, excludes air).

    Step breakdown on 512×512×769 @0.9mm scan:
      Load: ~9s (I/O bottleneck, gzip decompress)
      Orient+CropFg: ~213ms on full vol
      RandNativePatchd: 1ms
      ScaleIntensity: ~22ms on 7M voxels  (was 654ms on 200M)
      Spacingd: ~68ms on 7M voxels        (was 1670ms on 200M)
    """
    img_key = ["image"]
    target_mm = tuple(patch_size[i] * spacing[i] * 1.06 for i in range(3))  # 6% margin

    return Compose([
        LoadImaged(keys=img_key),
        EnsureChannelFirstd(keys=img_key),
        Orientationd(keys=img_key, axcodes="RAS"),
        # CropForeground on raw HU: > -500 keeps all tissue+bone+fat, excludes background air
        CropForegroundd(keys=img_key, source_key="image", select_fn=lambda x: x > -500),
        RandNativePatchd(keys=img_key, target_mm=target_mm),
        # ScaleIntensity AFTER crop: runs on small patch (~7M voxels)
        ScaleIntensityRanged(
            keys=img_key, a_min=hu_min, a_max=hu_max,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        Spacingd(keys=img_key, pixdim=list(spacing), mode="bilinear"),
        SpatialPadd(keys=img_key, spatial_size=patch_size),
        CenterSpatialCropd(keys=img_key, roi_size=patch_size),
        FastAugd(keys=img_key, flip_prob=0.5, rot90_prob=0.3),
        *_intensity_aug_ct(img_key),
        EnsureTyped(keys=img_key, dtype="float32"),
    ])


def _ct_val_cropfirst(
    patch_size: Tuple[int, int, int],
    spacing: Tuple[float, float, float],
    hu_min: float,
    hu_max: float,
) -> Compose:
    """Validación con Option-A (determinista: crop central en espacio nativo)."""
    img_key = ["image"]
    target_mm = tuple(patch_size[i] * spacing[i] * 1.06 for i in range(3))

    return Compose([
        LoadImaged(keys=img_key),
        EnsureChannelFirstd(keys=img_key),
        Orientationd(keys=img_key, axcodes="RAS"),
        CropForegroundd(keys=img_key, source_key="image", select_fn=lambda x: x > -500),
        RandNativePatchd(keys=img_key, target_mm=target_mm, center=True),
        ScaleIntensityRanged(
            keys=img_key, a_min=hu_min, a_max=hu_max,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        Spacingd(keys=img_key, pixdim=list(spacing), mode="bilinear"),
        SpatialPadd(keys=img_key, spatial_size=patch_size),
        CenterSpatialCropd(keys=img_key, roi_size=patch_size),
        EnsureTyped(keys=img_key, dtype="float32"),
    ])


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def build_ct_transforms(
    mode: str,
    patch_size: Tuple[int, int, int] = (96, 96, 96),
    spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5),
    hu_min: float = -1000.0,
    hu_max: float = 1000.0,
    fast: bool = False,
    cropfirst: bool = False,
) -> Compose:
    """
    Devuelve el pipeline de transforms correcto para CT.

    Args:
        mode:       "pretrain" | "classify" | "val"
        patch_size: Tamaño del parche 3D (D, H, W).
        spacing:    Spacing objetivo en mm.
        hu_min:     Límite inferior de la ventana HU.
        hu_max:     Límite superior de la ventana HU.
        fast:       Si True, usa pipeline .npy (requiere preprocess_ct.py). ~20ms/sample.
        cropfirst:  Si True, usa Option-A: crop nativo → resample parche. ~110ms/sample.
                    No requiere preprocesado offline. Auto-detectado en FlareCTDataset.
    """
    if fast:
        return _ct_pretrain_fast(patch_size) if mode == "pretrain" else _ct_val_fast(patch_size)

    if cropfirst:
        if mode == "pretrain":
            return _ct_pretrain_cropfirst(patch_size, spacing, hu_min, hu_max)
        else:
            return _ct_val_cropfirst(patch_size, spacing, hu_min, hu_max)

    if mode == "pretrain":
        return _ct_pretrain_transforms(patch_size, spacing, hu_min, hu_max)
    elif mode == "classify":
        return _ct_classify_transforms(patch_size, spacing, hu_min, hu_max)
    elif mode == "val":
        return _ct_val_transforms(patch_size, spacing, hu_min, hu_max)
    else:
        raise ValueError(f"Modo desconocido: '{mode}'. Usa 'pretrain', 'classify' o 'val'.")
