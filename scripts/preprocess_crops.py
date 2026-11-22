"""
Offline pre-extraction of (128, 128, 128) crops @ 1.5mm for fast training.

Why:
    - Loading .nii.gz: 1-9s per volume (gzip decompress, single-threaded)
    - Loading .npy crop:  ~2ms per crop  → 95%+ GPU utilization
    - Without this script: GPU idle ~95% of the time due to I/O

What it does:
    For each .nii.gz volume in train_part1/ and train_part2/:
      1. Loads once (expensive I/O)
      2. Orients to RAS + CropForeground (in raw HU)
      3. Extracts N random crops → resamples → raw HU int16 (NOT normalized)
      4. Saves as <dst>/train_part{1,2}/<case_id>_crop{i:02d}.npy

    The raw HU values allow applying multi-windowing (Pillar-0) at runtime:
    soft tissue [-150,250], lung [-1350,150], bone [-500,1300], mediastinum [-125,225]

Output: ~975GB for N=10 crops × 25,005 volumes × ~4.2MB per crop (int16)

Usage:
    uv run python scripts/preprocess_crops.py \\
        --src $LEJEPA_DATA_ROOT/flare_ct_pretrain_full \\
        --dst $LEJEPA_DATA_ROOT/flare_ct_pretrain_full_npy \\
        --n_crops 10 --workers 16

    # Dry-run: only counts files and estimates space
    uv run python scripts/preprocess_crops.py --src ... --dst ... --dry_run
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    SpatialPadd,
    Spacingd,
)

# Global crop configuration
CROP_SIZE = (128, 128, 128)     # (D, H, W) in voxels @ SPACING, 192mm isotropic FOV
SPACING   = (1.5, 1.5, 1.5)    # isotropic mm
# HU_MIN/HU_MAX only for CropForeground (> -500), NOT normalized on disk
# The .npy files are saved as raw int16 HU to allow multi-windowing at runtime

# target_mm with a 6% margin → after Spacingd it ends up slightly > CROP_SIZE
# SpatialPadd + CenterSpatialCropd adjust to the exact size
TARGET_MM = tuple(CROP_SIZE[i] * SPACING[i] * 1.06 for i in range(3))


def _build_load_tfm() -> Compose:
    """Loading pipeline: Load → Orient → CropForeground (in raw HU)."""
    keys = ["image"]
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS", labels=None),
        # CropFg in raw HU: > -500 keeps tissue + bone + fat, excludes air
        CropForegroundd(keys=keys, source_key="image", select_fn=lambda x: x > -500),
    ])


def _build_crop_tfm() -> Compose:
    """Crop pipeline: RandNativePatch → Spacingd → Pad → Crop → int16 HU raw.

    NOT normalized: we save raw HU (int16) for multi-windowing at runtime.
    int16 covers [-32768, 32767], enough for clinical HU [-1024, 3071].
    """
    # Import from data/; the worker needs the path in sys.path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.ct_transforms import RandNativePatchd

    keys = ["image"]
    return Compose([
        RandNativePatchd(keys=keys, target_mm=TARGET_MM, center=False),
        Spacingd(keys=keys, pixdim=list(SPACING), mode="bilinear"),
        SpatialPadd(keys=keys, spatial_size=CROP_SIZE),
        CenterSpatialCropd(keys=keys, roi_size=CROP_SIZE),
        EnsureTyped(keys=keys, dtype="int16"),
    ])


def _process_one(args: tuple) -> dict:
    """Worker: loads a volume and extracts n_crops random crops."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    src_path, dst_dir, n_crops = args
    t0 = time.perf_counter()

    # Check how many crops already exist (allows resuming)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    stem = src_path.stem.replace(".nii", "")
    existing = sum(1 for i in range(n_crops) if (dst_dir / f"{stem}_crop{i:02d}.npy").exists())
    if existing == n_crops:
        return {"path": str(src_path), "status": "skip", "n_crops": n_crops}

    try:
        # Expensive load (once per volume)
        load_tfm = _build_load_tfm()
        data = load_tfm({"image": str(src_path)})

        # Extract n_crops crops (reusing the loaded volume)
        crop_tfm = _build_crop_tfm()
        shapes = []
        for i in range(n_crops):
            dst_path = dst_dir / f"{stem}_crop{i:02d}.npy"
            if dst_path.exists():
                continue
            result = crop_tfm({"image": data["image"]})
            arr = result["image"].numpy().squeeze(0).astype(np.int16)    # (D, H, W) HU raw
            np.save(str(dst_path), arr)
            shapes.append(list(arr.shape))

        elapsed = time.perf_counter() - t0
        return {
            "path": str(src_path),
            "status": "ok",
            "n_crops": n_crops,
            "shapes": shapes,
            "elapsed_s": round(elapsed, 2),
        }
    except Exception as e:
        return {"path": str(src_path), "status": "error", "error": str(e)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src",     required=True,  help="Root with train_part1/ and train_part2/")
    ap.add_argument("--dst",     required=True,  help="Destination for .npy crops")
    ap.add_argument("--n_crops", type=int, default=10, help="Crops per volume")
    _slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
    _available_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 16)
    # Without SLURM, sched_getaffinity returns all CPUs of the node → conservative cap
    _default_workers = _slurm_cpus if _slurm_cpus > 0 else min(_available_cpus, 8)
    ap.add_argument("--workers", type=int, default=_default_workers)
    ap.add_argument("--subdirs", nargs="+", default=["train_part1", "train_part2"])
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    _cap = _slurm_cpus if _slurm_cpus > 0 else _available_cpus
    args.workers = min(args.workers, _cap)
    print(f"Workers: {args.workers} (CPUs available: {_available_cpus}, SLURM: {_slurm_cpus or 'N/A'})")

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    # Discover all .nii.gz in the training subdirectories
    all_nii: List[Path] = []
    for sub in args.subdirs:
        subdir = src / sub
        if not subdir.exists():
            print(f"WARN: {subdir} does not exist, skipping")
            continue
        found = sorted(p for p in subdir.glob("*.nii.gz") if not p.name.startswith("._"))
        all_nii.extend(found)
        print(f"  {sub}: {len(found)} files")

    print(f"\nTotal: {len(all_nii)} volumes")

    if args.dry_run:
        total_gb = sum(p.stat().st_size for p in all_nii) / 1e9
        bytes_per_crop = np.prod(CROP_SIZE) * 2  # int16 = 2 bytes
        out_gb = len(all_nii) * args.n_crops * bytes_per_crop / 1e9
        print(f"Input size: {total_gb:.1f} GB (compressed)")
        print(f"Estimated output: {out_gb:.1f} GB ({args.n_crops} crops × {len(all_nii)} vols × {bytes_per_crop/1e6:.1f}MB, int16 HU raw)")
        print(f"Crop size: {CROP_SIZE} @ {SPACING}mm → {TARGET_MM[0]:.0f}×{TARGET_MM[1]:.0f}×{TARGET_MM[2]:.0f}mm of native volume")
        return

    # Build tasks
    tasks = []
    for p in all_nii:
        sub_name = p.parent.name  # train_part1 or train_part2
        dst_dir  = dst / sub_name
        tasks.append((p, dst_dir, args.n_crops))

    # Count how many are already complete
    already_done = 0
    for p, dst_dir, n in tasks:
        stem = p.stem.replace(".nii", "")
        if all((Path(dst_dir) / f"{stem}_crop{i:02d}.npy").exists() for i in range(n)):
            already_done += 1
    print(f"Already complete: {already_done}/{len(tasks)}, pending: {len(tasks)-already_done}")
    print(f"Extracting {args.n_crops} crops × {len(tasks)} volumes with {args.workers} workers...\n")

    results = []
    t_start = time.perf_counter()
    with mp.Pool(processes=args.workers) as pool:
        pbar = tqdm(
            pool.imap_unordered(_process_one, tasks, chunksize=1),
            total=len(tasks),
            desc="Crops",
            unit="vol",
            dynamic_ncols=True,
        )
        for res in pbar:
            results.append(res)
            ok_so_far   = sum(1 for r in results if r["status"] == "ok")
            skip_so_far = sum(1 for r in results if r["status"] == "skip")
            err_so_far  = sum(1 for r in results if r["status"] == "error")
            elapsed     = time.perf_counter() - t_start
            rate        = len(results) / max(elapsed, 1e-9)
            eta_min     = (len(tasks) - len(results)) / max(rate, 1e-9) / 60
            pbar.set_postfix(
                ok=ok_so_far, skip=skip_so_far, err=err_so_far,
                rate=f"{rate:.1f}v/s", eta=f"{eta_min:.0f}m",
                refresh=False,
            )
            if res["status"] == "error":
                pbar.write(f"  ERROR: {res['path']} — {res['error']}")

    ok    = sum(1 for r in results if r["status"] == "ok")
    skip  = sum(1 for r in results if r["status"] == "skip")
    error = sum(1 for r in results if r["status"] == "error")
    total_t = time.perf_counter() - t_start
    total_crops = (ok + skip) * args.n_crops
    print(f"\nListo: {ok} nuevos + {skip} already existed + {error} errors in {total_t/60:.1f} min")
    print(f"Total crops saved: ~{total_crops:,}  ({total_crops * np.prod(CROP_SIZE) * 2 / 1e9:.1f} GB)")

    # Manifest for FlareCTDataset (fast mode)
    manifest = {
        "src": str(src),
        "crop_size": list(CROP_SIZE),
        "spacing": list(SPACING),
        "dtype": "int16",
        "normalized": False,
        "note": "HU raw — apply multi-windowing at runtime (Pillar-0)",
        "n_crops_per_vol": args.n_crops,
        "n_volumes": ok + skip,
        "total_crops": total_crops,
    }
    with open(dst / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {dst / 'manifest.json'}")


if __name__ == "__main__":
    main()
