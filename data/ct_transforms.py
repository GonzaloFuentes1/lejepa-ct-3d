"""
MONAI transforms for 3D CT (FLARE-Task4-CT-FM and CVPR26 downstream).

CT normalization:
    - HU windowing: clip to [hu_min, hu_max] then rescale to [0, 1]
    - Default: [-1000, 1000] HU (covers air, fat, tissue, bone)
    - Multi-windowing (Pillar-0): 4 channels with specialized windows
      soft tissue [-150,250], lung [-1350,150], bone [-500,1300], mediastinum [-125,225]
    - Resampling to isotropic spacing (1.5 mm by default)
    - Canonical RAS orientation

Differences from MRI (mri_transforms.py):
    - 1 channel (not multichannel like BraTS)
    - ScaleIntensityRanged instead of NormalizeIntensityd (HU has an absolute scale)
    - Coarser spacing (1.5 mm) for whole-body CT
    - Patch size 96³ (larger CT volumes, limited memory)
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


# HU windows for multi-windowing (Pillar-0, Table 2)
# Each tuple: (hu_min, hu_max)
PILLAR0_WINDOWS = [
    (-150,   250),   # soft tissue
    (-1350,  150),   # lung
    (-500,  1300),   # bone
    (-125,   225),   # mediastinum
]


class LoadNpyd(MapTransform):
    """
    Loads a preprocessed volume from .npy (int16 HU raw) → float32 4-channel tensor.

    Applies Pillar-0 multi-windowing at runtime: converts raw HU to 4 channels
    normalized to [0,1] with specialized windows (soft tissue, lung, bone, mediastinum).

    Input:  .npy int16 (D, H, W) HU raw
    Output: tensor float16 (4, D, H, W) normalized to [0,1] per channel
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
    Applies Pillar-0 multi-windowing to an already loaded raw HU tensor.

    For the slow/cropfirst pipeline: converts (1, D, H, W) HU raw
    to (4, D, H, W) float32 multi-window.
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
    Fast spatial augmentations in pure PyTorch (no MetaTensor overhead).

    For the fast .npy pipeline: ~3ms vs ~20ms with MONAI RandFlipd/Rotate90d.
    Operates on (1, D, H, W) float16 or float32 tensors.
    """

    def __init__(self, keys, flip_prob: float = 0.5, rot90_prob: float = 0.3):
        super().__init__(keys)
        self.flip_prob  = flip_prob
        self.rot90_prob = rot90_prob

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.keys:
            x = d[key]  # (1, D, H, W)
            # Random flips over the 3 spatial axes
            for axis in (1, 2, 3):
                if torch.rand(1).item() < self.flip_prob:
                    x = x.flip(axis)
            # Random 90° rotations only in the axial in-plane (H-W, dims 2-3).
            # Do not rotate in planes involving D (dims 1-2 or 1-3): with non-cubic
            # crops (D=96 ≠ H=W=160) rot90 would swap D↔H or D↔W and produce
            # tensors of different shapes that cannot be stacked in the batch.
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
# Base CT transforms
# ---------------------------------------------------------------------------

def _ct_base_transforms(
    img_keys: List[str],
    spacing: Tuple[float, float, float],
    hu_min: float,
    hu_max: float,
    extra_keys: List[str] = [],
) -> List:
    """
    Base pipeline for any CT volume:
      1. Load NIfTI
      2. Canonical RAS orientation
      3. Resampling to the target spacing
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
        # HU windowing: clip + rescale to [0, 1]
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
# Standard augmentations
# ---------------------------------------------------------------------------

def _spatial_aug_ct(keys: List[str], label_keys: List[str] = []) -> List:
    """Spatial augmentations for CT: flips + 90° rotations + zoom."""
    return [
        RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
        # 90° rotations: free (axis permutation, no interpolation)
        # They force orientation invariance; CT from different centers varies
        RandRotate90d(keys=keys, prob=0.3, spatial_axes=(0, 1)),
        RandRotate90d(keys=keys, prob=0.2, spatial_axes=(0, 2)),
        # Conservative zoom with keep_size=True to keep 96³ fixed
        # (without this, the output size varies and the ViT3D 6×6×6 grid fails)
        RandZoomd(keys=keys, min_zoom=0.85, max_zoom=1.15, prob=0.3, keep_size=True),
    ]


def _intensity_aug_ct(img_keys: List[str]) -> List:
    """Light intensity augmentations for CT (HU range already fixed)."""
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
# Pipelines per mode
# ---------------------------------------------------------------------------

def _ct_pretrain_transforms(
    patch_size: Tuple[int, int, int],
    spacing: Tuple[float, float, float],
    hu_min: float,
    hu_max: float,
) -> Compose:
    """SSL pretrain: image only, random crop, augmentations."""
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
    Classification (linear probe / fine-tune): full image without crop,
    since we need a global representation of the volume.

    The encoder extracts features from the full volume → global pool → linear.
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
    """Validation/test: no augmentations, deterministic center crop."""
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
# Fast pipelines (for preprocessed .npy: no Load/Orient/Resample)
# ---------------------------------------------------------------------------

def _ct_pretrain_fast(patch_size: Tuple[int, int, int]) -> Compose:
    """
    Fast pipeline for preprocessed .npy (int16 HU raw).

    LoadNpyd applies Pillar-0 multi-windowing at runtime: int16 HU raw
    → 4 canales float16 [0,1] (soft tissue, lung, bone, mediastinum).

    Uses FastAugd (pure PyTorch) instead of MONAI for the spatial
    augmentations: avoids MetaTensor overhead (~20ms → ~3ms).

    Speedup vs original pipeline: ~900ms → ~35ms per sample.
    """
    img_key = ["image"]
    return Compose([
        LoadNpyd(keys=img_key),                                         # int16 HU raw → (4,D,H,W) float16
        SpatialPadd(keys=img_key, spatial_size=patch_size),             # pad if vol < patch_size
        RandSpatialCropd(keys=img_key, roi_size=patch_size, random_size=False),
        FastAugd(keys=img_key, flip_prob=0.5, rot90_prob=0.3),         # fast flips + rot90
        *_intensity_aug_ct(img_key),                                    # MONAI intensity augs
        EnsureTyped(keys=img_key, dtype="float32"),                     # float16 → float32
    ])


def _ct_val_fast(patch_size: Tuple[int, int, int]) -> Compose:
    """Fast validation pipeline for .npy (int16 HU raw → 4 channels)."""
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
    """Validation with Option-A (deterministic: center crop in native space)."""
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
# Public API
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
    Returns the right transforms pipeline for CT.

    Args:
        mode:       "pretrain" | "classify" | "val"
        patch_size: 3D patch size (D, H, W).
        spacing:    Target spacing in mm.
        hu_min:     Lower bound of the HU window.
        hu_max:     Upper bound of the HU window.
        fast:       If True, uses the .npy pipeline (requires preprocess_crops.py). ~20ms/sample.
        cropfirst:  If True, uses Option-A: native crop → resample patch. ~110ms/sample.
                    Needs no offline preprocessing. Auto-detected in FlareCTDataset.
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
        raise ValueError(f"Unknown mode: '{mode}'. Use 'pretrain', 'classify' or 'val'.")
