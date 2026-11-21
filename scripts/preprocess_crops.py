"""
Pre-extracción offline de crops (128, 128, 128) @ 1.5mm para entrenamiento rápido.

Por qué:
    - Cargar .nii.gz: 1–9s por volumen (gzip decompress, single-threaded)
    - Cargar .npy crop:  ~2ms por crop  → GPU al 95%+ de utilización
    - Sin este script: GPU ocioso ~95% del tiempo por I/O

Qué hace:
    Por cada volumen .nii.gz en train_part1/ y train_part2/:
      1. Carga una sola vez (I/O caro)
      2. Orienta a RAS + CropForeground (en HU raw)
      3. Extrae N crops aleatorios → resamplea → HU raw int16 (SIN normalizar)
      4. Guarda como <dst>/train_part{1,2}/<case_id>_crop{i:02d}.npy

    Los valores HU raw permiten aplicar multi-windowing (Pillar-0) en runtime:
    soft tissue [-150,250], lung [-1350,150], bone [-500,1300], mediastinum [-125,225]

Output: ~975GB para N=10 crops × 25,005 volúmenes × ~4.2MB por crop (int16)

Uso:
    uv run python scripts/preprocess_crops.py \\
        --src $LEJEPA_DATA_ROOT/flare_ct_pretrain_full \\
        --dst $LEJEPA_DATA_ROOT/flare_ct_pretrain_full_npy \\
        --n_crops 10 --workers 16

    # Dry-run: solo cuenta archivos y estima espacio
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

# Configuración global del crop
CROP_SIZE = (128, 128, 128)     # (D, H, W) en voxels @ SPACING — 192mm FOV isotrópico
SPACING   = (1.5, 1.5, 1.5)    # mm isotrópico
# HU_MIN/HU_MAX solo para CropForeground (> -500), NO se normaliza en disco
# Los .npy se guardan como int16 HU raw para permitir multi-windowing en runtime

# target_mm con 6% de margen → después de Spacingd queda ligeramente > CROP_SIZE
# SpatialPadd + CenterSpatialCropd ajustan al tamaño exacto
TARGET_MM = tuple(CROP_SIZE[i] * SPACING[i] * 1.06 for i in range(3))


def _build_load_tfm() -> Compose:
    """Pipeline de carga: Load → Orient → CropForeground (en HU raw)."""
    keys = ["image"]
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS", labels=None),
        # CropFg en HU raw: > -500 conserva tejido + hueso + grasa, excluye aire
        CropForegroundd(keys=keys, source_key="image", select_fn=lambda x: x > -500),
    ])


def _build_crop_tfm() -> Compose:
    """Pipeline de crop: RandNativePatch → Spacingd → Pad → Crop → int16 HU raw.

    NO se normaliza: guardamos HU raw (int16) para multi-windowing en runtime.
    int16 cubre [-32768, 32767] — suficiente para HU clínico [-1024, 3071].
    """
    # Importar desde data/ — el worker necesita el path en sys.path
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
    """Worker: carga un volumen y extrae n_crops crops aleatorios."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    src_path, dst_dir, n_crops = args
    t0 = time.perf_counter()

    # Verificar cuántos crops ya existen (permite reanudar)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    stem = src_path.stem.replace(".nii", "")
    existing = sum(1 for i in range(n_crops) if (dst_dir / f"{stem}_crop{i:02d}.npy").exists())
    if existing == n_crops:
        return {"path": str(src_path), "status": "skip", "n_crops": n_crops}

    try:
        # Carga costosa (1 vez por volumen)
        load_tfm = _build_load_tfm()
        data = load_tfm({"image": str(src_path)})

        # Extrae n_crops crops (reutilizando el volumen cargado)
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
    ap.add_argument("--src",     required=True,  help="Root con train_part1/ y train_part2/")
    ap.add_argument("--dst",     required=True,  help="Destino para crops .npy")
    ap.add_argument("--n_crops", type=int, default=10, help="Crops por volumen")
    _slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
    _available_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 16)
    # Sin SLURM, sched_getaffinity devuelve todos los CPUs del nodo → cap conservador
    _default_workers = _slurm_cpus if _slurm_cpus > 0 else min(_available_cpus, 8)
    ap.add_argument("--workers", type=int, default=_default_workers)
    ap.add_argument("--subdirs", nargs="+", default=["train_part1", "train_part2"])
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    _cap = _slurm_cpus if _slurm_cpus > 0 else _available_cpus
    args.workers = min(args.workers, _cap)
    print(f"Workers: {args.workers} (CPUs disponibles: {_available_cpus}, SLURM: {_slurm_cpus or 'N/A'})")

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    # Descubrir todos los .nii.gz en los subdirectorios de entrenamiento
    all_nii: List[Path] = []
    for sub in args.subdirs:
        subdir = src / sub
        if not subdir.exists():
            print(f"WARN: {subdir} no existe, saltando")
            continue
        found = sorted(p for p in subdir.glob("*.nii.gz") if not p.name.startswith("._"))
        all_nii.extend(found)
        print(f"  {sub}: {len(found)} archivos")

    print(f"\nTotal: {len(all_nii)} volúmenes")

    if args.dry_run:
        total_gb = sum(p.stat().st_size for p in all_nii) / 1e9
        bytes_per_crop = np.prod(CROP_SIZE) * 2  # int16 = 2 bytes
        out_gb = len(all_nii) * args.n_crops * bytes_per_crop / 1e9
        print(f"Input size: {total_gb:.1f} GB (comprimido)")
        print(f"Output estimado: {out_gb:.1f} GB ({args.n_crops} crops × {len(all_nii)} vols × {bytes_per_crop/1e6:.1f}MB, int16 HU raw)")
        print(f"Crop size: {CROP_SIZE} @ {SPACING}mm → {TARGET_MM[0]:.0f}×{TARGET_MM[1]:.0f}×{TARGET_MM[2]:.0f}mm de volumen nativo")
        return

    # Construir tareas
    tasks = []
    for p in all_nii:
        sub_name = p.parent.name  # train_part1 o train_part2
        dst_dir  = dst / sub_name
        tasks.append((p, dst_dir, args.n_crops))

    # Contar cuántos ya están completos
    already_done = 0
    for p, dst_dir, n in tasks:
        stem = p.stem.replace(".nii", "")
        if all((Path(dst_dir) / f"{stem}_crop{i:02d}.npy").exists() for i in range(n)):
            already_done += 1
    print(f"Ya completos: {already_done}/{len(tasks)}, pendientes: {len(tasks)-already_done}")
    print(f"Extrayendo {args.n_crops} crops × {len(tasks)} volúmenes con {args.workers} workers...\n")

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
    print(f"\nListo: {ok} nuevos + {skip} ya existían + {error} errores en {total_t/60:.1f} min")
    print(f"Total crops guardados: ~{total_crops:,}  ({total_crops * np.prod(CROP_SIZE) * 2 / 1e9:.1f} GB)")

    # Manifest para FlareCTDataset (modo fast)
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
