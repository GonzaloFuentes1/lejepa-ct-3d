"""
Inspection of a CT volume from the CVPR26 challenge.

Loads a CT NIfTI volume, applies HU windowing, and saves:
  - Axial/coronal/sagittal slices as PNG
  - Histogram of HU values before and after windowing
  - Basic volume statistics (shape, spacing, HU stats)

Usage:
    cd .
    python examples/ct/inspect_ct_volume.py --nifti /path/to/volume.nii.gz
    python examples/ct/inspect_ct_volume.py --auto   # uses the first available volume
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import nibabel as nib
except ImportError:
    print("Install nibabel: uv pip install nibabel")
    sys.exit(1)

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True, parents=True)

# Challenge HU windowing
HU_MIN = -1000.0
HU_MAX  =  1000.0

# Colores
plt.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor":   "#1e293b",
    "axes.edgecolor":   "#334155",
    "axes.labelcolor":  "#cbd5e1",
    "xtick.color":      "#94a3b8",
    "ytick.color":      "#94a3b8",
    "text.color":       "#f1f5f9",
    "grid.color":       "#334155",
    "grid.alpha":       0.5,
    "axes.titlecolor":  "#f1f5f9",
})

ACCENT = "#38bdf8"
ACCENT2 = "#34d399"


def load_nifti_volume(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Loads a NIfTI and returns (HxWxD float32 array, voxel_spacing)."""
    img = nib.load(str(path))
    vol = img.get_fdata(dtype=np.float32)
    zooms = np.abs(np.array(img.header.get_zooms()[:3], dtype=np.float32))
    return vol, zooms


def hu_window(vol: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    """Clip HU range and rescale to [0, 1]."""
    clipped = np.clip(vol, hu_min, hu_max)
    return (clipped - hu_min) / (hu_max - hu_min)


def plot_slices(vol_raw: np.ndarray, vol_norm: np.ndarray, path: Path, title: str) -> None:
    """3 planos ortogonales: raw HU vs windowed."""
    D, H, W = vol_raw.shape
    mid_d, mid_h, mid_w = D // 2, H // 2, W // 2

    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    fig.suptitle(title, fontsize=13, fontweight="bold", color="#f1f5f9")
    gs = gridspec.GridSpec(2, 3, figure=fig)

    slices_raw  = [vol_raw[mid_d, :, :], vol_raw[:, mid_h, :], vol_raw[:, :, mid_w]]
    slices_norm = [vol_norm[mid_d, :, :], vol_norm[:, mid_h, :], vol_norm[:, :, mid_w]]
    plane_names = ["Axial", "Coronal", "Sagital"]

    for col, (s_raw, s_norm, name) in enumerate(zip(slices_raw, slices_norm, plane_names)):
        # Fila 0: raw HU
        ax0 = fig.add_subplot(gs[0, col])
        im0 = ax0.imshow(s_raw.T, cmap="gray", origin="lower",
                         vmin=HU_MIN, vmax=HU_MAX, aspect="auto")
        ax0.set_title(f"{name} — HU raw", fontsize=10)
        ax0.axis("off")
        plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

        # Fila 1: windowed [0,1]
        ax1 = fig.add_subplot(gs[1, col])
        ax1.imshow(s_norm.T, cmap="gray", origin="lower",
                   vmin=0, vmax=1, aspect="auto")
        ax1.set_title(f"{name} — windowed [0,1]", fontsize=10)
        ax1.axis("off")

    out_path = path / "slices.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  Slices saved: {out_path}")


def plot_histogram(vol_raw: np.ndarray, path: Path) -> None:
    """HU histogram: full vs useful range."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.patch.set_facecolor("#0f172a")

    for ax in axes:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8")
        ax.spines[:].set_color("#334155")

    flat = vol_raw.flatten()

    # HU completo
    axes[0].hist(flat, bins=256, color=ACCENT, alpha=0.85, edgecolor="none")
    axes[0].set_title("Full HU distribution", fontsize=11)
    axes[0].set_xlabel("HU value")
    axes[0].set_ylabel("Voxels")
    axes[0].axvline(HU_MIN, color="#f87171", linestyle="--", linewidth=1.5, label=f"HU_MIN={HU_MIN:.0f}")
    axes[0].axvline(HU_MAX, color="#fb923c", linestyle="--", linewidth=1.5, label=f"HU_MAX={HU_MAX:.0f}")
    axes[0].legend(fontsize=9)

    # Windowing range only
    windowed_flat = flat[(flat >= HU_MIN) & (flat <= HU_MAX)]
    axes[1].hist(windowed_flat, bins=256, color=ACCENT2, alpha=0.85, edgecolor="none")
    axes[1].set_title(f"HU in [{HU_MIN:.0f}, {HU_MAX:.0f}] (windowing)", fontsize=11)
    axes[1].set_xlabel("HU value")

    fig.tight_layout()
    out_path = path / "histogram.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  Histogram saved: {out_path}")


def print_stats(vol: np.ndarray, spacing: np.ndarray, path: Path) -> None:
    """Prints and saves volume statistics."""
    stats = {
        "shape":        list(vol.shape),
        "spacing_mm":   spacing.tolist(),
        "fov_mm":       (np.array(vol.shape) * spacing).tolist(),
        "hu_min":       float(vol.min()),
        "hu_max":       float(vol.max()),
        "hu_mean":      float(vol.mean()),
        "hu_std":       float(vol.std()),
        "hu_p1":        float(np.percentile(vol, 1)),
        "hu_p99":       float(np.percentile(vol, 99)),
        "n_voxels":     int(vol.size),
        "nonzero_pct":  float(100 * np.count_nonzero(vol) / vol.size),
    }

    print()
    print("  CT volume statistics:")
    print(f"    Shape:          {stats['shape']}")
    print(f"    Spacing (mm):   {[f'{s:.2f}' for s in stats['spacing_mm']]}")
    print(f"    FOV (mm):       {[f'{v:.1f}' for v in stats['fov_mm']]}")
    print(f"    HU range:       [{stats['hu_min']:.1f}, {stats['hu_max']:.1f}]")
    print(f"    HU mean±std:    {stats['hu_mean']:.1f} ± {stats['hu_std']:.1f}")
    print(f"    HU p1/p99:      {stats['hu_p1']:.1f} / {stats['hu_p99']:.1f}")
    print(f"    Non-zero:       {stats['nonzero_pct']:.1f}%")

    import json
    stats_path = path / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  ✓  Stats saved: {stats_path}")


def find_first_volume(data_root: Path) -> Path | None:
    """Finds the first .nii.gz in the downstream dataset structure."""
    search_dirs = [
        data_root / "cvpr26_downstream",
        data_root / "flare_ct_pretrain" / "volumes",
    ]
    for d in search_dirs:
        if d.exists():
            vols = sorted(d.rglob("*.nii.gz"))
            if vols:
                return vols[0]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a CT volume (HU windowing, slices, histogram)",
    )
    parser.add_argument(
        "--nifti", type=str, default=None,
        help="Path to the .nii.gz file to inspect",
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="Automatically find the first available volume",
    )
    parser.add_argument(
        "--data_root", type=str, default=os.environ.get("LEJEPA_DATA_ROOT", "./data"),
        help="Data root (for --auto)",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output directory for the images (default: examples/ct/outputs/)",
    )
    args = parser.parse_args()

    # Resolve the volume path
    if args.nifti:
        nifti_path = Path(args.nifti)
    elif args.auto:
        nifti_path = find_first_volume(Path(args.data_root))
        if nifti_path is None:
            print("No .nii.gz volume found in data_root.")
            print("Download the data first:")
            print("  python scripts/download_ct_datasets.py --downstream")
            sys.exit(1)
        print(f"  → Using: {nifti_path}")
    else:
        parser.print_help()
        print("\n  Specify --nifti <path> or use --auto")
        sys.exit(0)

    if not nifti_path.exists():
        print(f"File not found: {nifti_path}")
        sys.exit(1)

    out_dir = Path(args.out) if args.out else OUT / nifti_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nInspecting: {nifti_path.name}")
    print(f"Output in:      {out_dir}")

    # Load and process
    vol_raw, spacing = load_nifti_volume(nifti_path)
    vol_norm = hu_window(vol_raw, HU_MIN, HU_MAX)

    # Generar outputs
    print_stats(vol_raw, spacing, out_dir)
    plot_slices(vol_raw, vol_norm, out_dir, title=nifti_path.name)
    plot_histogram(vol_raw, out_dir)

    print(f"\n  ✓  Inspection complete → {out_dir}/\n")


if __name__ == "__main__":
    main()
