"""
Local Linear Probing evaluation: replicates the CVPR 2026 competition pipeline.

Steps:
  1. Extracts embeddings of all AMOS volumes (cached in --feat_dir)
  2. For each task: LogisticRegression(train) → BalAcc + AUROC(val)
  3. Prints a comparison table against the competition baselines

Usage:
    # Extract features + evaluate LP (first time)
    python scripts/run_local_lp.py \\
        --ckpt checkpoints/best.ckpt \\
        --data_dir $LEJEPA_DATA_ROOT/cvpr26_downstream/AMOS-clf-tr-val \\
        --feat_dir $LEJEPA_DATA_ROOT/cvpr26_lp_feats \\
        --device cuda:0

    # Evaluate LP only (features already extracted)
    python scripts/run_local_lp.py \\
        --feat_dir $LEJEPA_DATA_ROOT/cvpr26_lp_feats \\
        --data_dir $LEJEPA_DATA_ROOT/cvpr26_downstream/AMOS-clf-tr-val
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import (
    Compose, EnsureChannelFirstd, EnsureTyped,
    LoadImaged, Orientationd, ScaleIntensityRanged, Spacingd,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

PATCH_SIZE = 16
SPACING    = (1.5, 1.5, 1.5)
HU_MIN     = -1000.0
HU_MAX     =  1000.0

TASKS = [
    "adrenal_hyperplasia", "ascites", "atherosclerosis", "cholecystitis",
    "colorectal_cancer", "fatty_liver", "gallstone", "hydronephrosis",
    "kidney_stone", "liver_calcifications", "liver_cyst", "liver_lesion",
    "lymphadenopathy", "renal_cyst", "splenomegaly",
]
ROI_TASKS = {
    "adrenal_hyperplasia", "cholecystitis", "fatty_liver", "gallstone",
    "hydronephrosis", "kidney_stone", "liver_calcifications", "liver_cyst",
    "liver_lesion", "renal_cyst", "splenomegaly",
}

# Reference baseline BalAcc (Curia, best public AMOS baseline)
CURIA_BALACC = {
    "adrenal_hyperplasia": 0.786, "ascites": 0.750, "atherosclerosis": 0.809,
    "cholecystitis": 0.755, "colorectal_cancer": 0.750, "fatty_liver": 0.949,
    "gallstone": 0.819, "hydronephrosis": 0.906, "kidney_stone": 0.677,
    "liver_calcifications": 0.667, "liver_cyst": 0.668, "liver_lesion": 0.694,
    "lymphadenopathy": 0.806, "renal_cyst": 0.688, "splenomegaly": 0.900,
}


def _build_transforms() -> Compose:
    keys = ["image"]
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=list(SPACING), mode="bilinear"),
        ScaleIntensityRanged(keys=keys, a_min=HU_MIN, a_max=HU_MAX,
                             b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=keys, dtype="float32"),
    ])


def _build_mask_transforms() -> Compose:
    keys = ["mask"]
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=list(SPACING), mode="nearest"),
        EnsureTyped(keys=keys, dtype="float32"),
    ])


@torch.no_grad()
def _extract_one(encoder, vol, mask, device) -> np.ndarray:
    """Encodes a volume → 1-D embedding (out_dim,)."""
    tokens = encoder(vol).squeeze(0)  # (N, embed_dim)
    roi_mask = None
    if mask is not None:
        m_down = F.avg_pool3d(mask.float(), kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
        roi_flat = (m_down.squeeze() > 0).reshape(-1)
        roi_mask = roi_flat.unsqueeze(0) if roi_flat.any() else None
    feat = encoder.project(tokens.unsqueeze(0), roi_mask=roi_mask)
    return feat.squeeze(0).cpu().float().numpy()


def extract_features(
    ckpt: str,
    data_dir: Path,
    feat_dir: Path,
    device: torch.device,
    fp16: bool = True,
) -> None:
    """Extracts embeddings for all AMOS volumes and saves them in feat_dir."""
    print(f"Loading checkpoint: {ckpt}")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.encoder.lejepa_module import LeJEPAModule

    module = LeJEPAModule.load_from_checkpoint(ckpt, map_location="cpu")
    encoder = module.encoder.to(device).eval()
    if fp16:
        encoder = encoder.half()
    del module
    torch.cuda.empty_cache()

    print(f"Encoder out_dim: {encoder.out_dim}")

    img_tfm  = _build_transforms()
    msk_tfm  = _build_mask_transforms()
    img_dir  = data_dir / "images"
    mask_root = data_dir / "fg_masks"

    cases = sorted(img_dir.glob("*.nii.gz"))
    print(f"Extracting features from {len(cases)} volumes...")

    feat_dir.mkdir(parents=True, exist_ok=True)

    # Extract per ROI task for correct masked pooling
    # First extract global (no ROI) for non-ROI tasks
    global_dir = feat_dir / "global"
    global_dir.mkdir(exist_ok=True)

    for i, case_path in enumerate(cases, 1):
        case_id = case_path.name.replace(".nii.gz", "")
        out_path = global_dir / f"{case_id}.h5"
        if out_path.exists():
            continue

        print(f"  [{i}/{len(cases)}] {case_id}", end="", flush=True)
        try:
            data = img_tfm({"image": str(case_path)})
            vol  = data["image"].unsqueeze(0).to(device)
            if fp16:
                vol = vol.half()

            feat = _extract_one(encoder, vol, None, device)
            with h5py.File(out_path, "w") as hf:
                hf.create_dataset("y_hat", data=feat, dtype="float32")
            print(f" {feat.shape} ✓")
        except Exception as e:
            print(f" ✗ {e}")

    # Extract for each ROI task with masked pooling
    for task in ROI_TASKS:
        task_mask_dir = mask_root / task
        if not task_mask_dir.exists():
            continue

        task_feat_dir = feat_dir / task
        task_feat_dir.mkdir(exist_ok=True)
        print(f"\nExtracting with ROI mask: {task}")

        for i, case_path in enumerate(cases, 1):
            case_id  = case_path.name.replace(".nii.gz", "")
            out_path = task_feat_dir / f"{case_id}.h5"
            if out_path.exists():
                continue

            mask_path = task_mask_dir / case_path.name
            if not mask_path.exists():
                # No mask → copy global
                global_h5 = global_dir / f"{case_id}.h5"
                if global_h5.exists():
                    import shutil
                    shutil.copy(global_h5, out_path)
                continue

            try:
                data = img_tfm({"image": str(case_path)})
                vol  = data["image"].unsqueeze(0).to(device)
                if fp16:
                    vol = vol.half()

                mdata = msk_tfm({"mask": str(mask_path)})
                mask_tensor = mdata["mask"].unsqueeze(0).to(device)
                if mask_tensor.shape[2:] != vol.shape[2:]:
                    mask_tensor = F.interpolate(mask_tensor, size=vol.shape[2:], mode="nearest")

                feat = _extract_one(encoder, vol, mask_tensor, device)
                with h5py.File(out_path, "w") as hf:
                    hf.create_dataset("y_hat", data=feat, dtype="float32")
            except Exception as e:
                print(f"  ✗ {case_id}: {e}")


def load_task_data(task: str, data_dir: Path, feat_dir: Path):
    """Loads (X_train, y_train, X_val, y_val) for a task."""
    label_csv = data_dir / "labels" / f"{task}.csv"
    feat_subdir = feat_dir / task if task in ROI_TASKS else feat_dir / "global"

    train_X, train_y, val_X, val_y = [], [], [], []

    with open(label_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = Path(row["case_id"]).stem.replace(".nii", "")
            label   = int(row[task])
            split   = row["split"]
            feat_path = feat_subdir / f"{case_id}.h5"

            if not feat_path.exists():
                # Fall back to global
                feat_path = feat_dir / "global" / f"{case_id}.h5"
            if not feat_path.exists():
                continue

            with h5py.File(feat_path, "r") as hf:
                feat = hf["y_hat"][:]

            if split == "train":
                train_X.append(feat)
                train_y.append(label)
            else:
                val_X.append(feat)
                val_y.append(label)

    return (
        np.stack(train_X), np.array(train_y),
        np.stack(val_X),   np.array(val_y),
    )


def run_lp(data_dir: Path, feat_dir: Path) -> dict:
    """Trains LP per task and returns metrics."""
    results = {}

    for task in TASKS:
        label_csv = data_dir / "labels" / f"{task}.csv"
        if not label_csv.exists():
            print(f"  SKIP {task}: CSV not found")
            continue

        try:
            X_tr, y_tr, X_val, y_val = load_task_data(task, data_dir, feat_dir)
            if len(X_tr) == 0 or len(X_val) == 0:
                print(f"  SKIP {task}: no features")
                continue

            scaler = StandardScaler()
            X_tr_s  = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)

            clf = LogisticRegression(
                max_iter=1000,
                C=0.1,
                class_weight="balanced",
                solver="lbfgs",
                random_state=42,
            )
            clf.fit(X_tr_s, y_tr)

            y_pred = clf.predict(X_val_s)
            y_prob = clf.predict_proba(X_val_s)[:, 1]

            bal_acc = balanced_accuracy_score(y_val, y_pred)
            try:
                auroc = roc_auc_score(y_val, y_prob)
            except ValueError:
                auroc = float("nan")

            results[task] = {"bal_acc": bal_acc, "auroc": auroc,
                             "n_train": len(y_tr), "n_val": len(y_val)}
        except Exception as e:
            print(f"  ERROR {task}: {e}")

    return results


def print_table(results: dict) -> None:
    """Prints a table comparing against Curia (best AMOS baseline)."""
    header = f"{'Task':<25} {'BalAcc':>7} {'AUROC':>7} {'Curia':>7} {'Δ':>7} {'n_tr':>6} {'n_val':>6}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    bal_accs, aurocs, deltas = [], [], []
    for task in TASKS:
        if task not in results:
            print(f"  {task:<23}  —")
            continue
        r   = results[task]
        ref = CURIA_BALACC.get(task, float("nan"))
        delta = r["bal_acc"] - ref
        sign  = "+" if delta >= 0 else ""
        print(
            f"  {task:<23} {r['bal_acc']:>7.4f} {r['auroc']:>7.4f} "
            f"{ref:>7.4f} {sign}{delta:>6.4f}  "
            f"{r['n_train']:>5} {r['n_val']:>5}"
        )
        bal_accs.append(r["bal_acc"])
        aurocs.append(r["auroc"])
        deltas.append(delta)

    if bal_accs:
        print("-" * len(header))
        mean_delta = np.mean(deltas)
        sign = "+" if mean_delta >= 0 else ""
        print(
            f"  {'MEAN':<23} {np.mean(bal_accs):>7.4f} {np.nanmean(aurocs):>7.4f} "
            f"{np.mean(list(CURIA_BALACC.values())):>7.4f} {sign}{mean_delta:>6.4f}"
        )
    print("=" * len(header))


def main() -> None:
    ap = argparse.ArgumentParser(description="Local LP eval — CVPR 2026 Task 1")
    ap.add_argument("--ckpt",     default=None, help="Checkpoint .ckpt (omit if features are already extracted)")
    ap.add_argument("--data_dir", required=True, help="AMOS-clf-tr-val root")
    ap.add_argument("--feat_dir", required=True, help="Dir to cache .h5 features")
    ap.add_argument("--device",   default="cuda:0")
    ap.add_argument("--fp16",     action="store_true", default=True)
    ap.add_argument("--skip_extract", action="store_true", help="Do not extract features, LP only")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    feat_dir = Path(args.feat_dir)
    device   = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if not args.skip_extract and args.ckpt:
        extract_features(args.ckpt, data_dir, feat_dir, device, fp16=args.fp16)
    elif not args.skip_extract and not args.ckpt:
        print("Pass --ckpt or --skip_extract if the features are already extracted.")
        return

    print("\nTraining Linear Probes...")
    results = run_lp(data_dir, feat_dir)
    print_table(results)


if __name__ == "__main__":
    main()
