"""
Linear Probing Evaluation Callback — CVPR 2026 AMOS downstream tasks.

Every `every_n_epochs` epochs:
  1. Extracts features of all AMOS volumes (train + val splits)
     using the current encoder (no gradient, in eval mode)
  2. Trains LogisticRegression on split=train per task
  3. Evaluates on split=val → BalAcc + AUROC
  4. Logs to W&B under section val/lp/

Runs on rank 0 only (DDP-safe).
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback

log = logging.getLogger(__name__)

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
CURIA_BALACC = {
    "adrenal_hyperplasia": 0.786, "ascites": 0.750, "atherosclerosis": 0.809,
    "cholecystitis": 0.755, "colorectal_cancer": 0.750, "fatty_liver": 0.949,
    "gallstone": 0.819, "hydronephrosis": 0.906, "kidney_stone": 0.677,
    "liver_calcifications": 0.667, "liver_cyst": 0.668, "liver_lesion": 0.694,
    "lymphadenopathy": 0.806, "renal_cyst": 0.688, "splenomegaly": 0.900,
}

PATCH_SIZE = 16
SPACING    = (1.5, 1.5, 1.5)
HU_MIN, HU_MAX = -1000.0, 1000.0


def _build_transforms():
    from monai.transforms import (
        Compose, EnsureChannelFirstd, EnsureTyped,
        LoadImaged, Orientationd, ScaleIntensityRanged, Spacingd,
    )
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=list(SPACING), mode="bilinear"),
        ScaleIntensityRanged(keys=["image"], a_min=HU_MIN, a_max=HU_MAX,
                             b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image"], dtype="float32"),
    ])


def _build_mask_transforms():
    from monai.transforms import (
        Compose, EnsureChannelFirstd, EnsureTyped,
        LoadImaged, Orientationd, Spacingd,
    )
    return Compose([
        LoadImaged(keys=["mask"]),
        EnsureChannelFirstd(keys=["mask"]),
        Orientationd(keys=["mask"], axcodes="RAS"),
        Spacingd(keys=["mask"], pixdim=list(SPACING), mode="nearest"),
        EnsureTyped(keys=["mask"], dtype="float32"),
    ])


def _load_labels(label_dir: Path) -> Dict[str, Dict[str, tuple]]:
    """
    Returns {task: {case_id: (label, split)}} for all available tasks.
    """
    data: Dict[str, Dict[str, tuple]] = {}
    for task in TASKS:
        csv_path = label_dir / f"{task}.csv"
        if not csv_path.exists():
            continue
        data[task] = {}
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                case_id = row["case_id"].replace(".nii.gz", "")
                label   = int(row[task])
                split   = row["split"].strip()
                data[task][case_id] = (label, split)
    return data


class LPEvalCallback(Callback):
    """
    Evaluates linear probing on the 15 AMOS tasks every `every_n_epochs` epochs.

    Args:
        data_dir:        Path to AMOS-clf-tr-val/ (contains images/, labels/, fg_masks/)
        every_n_epochs:  Evaluation frequency in epochs (default 3)
        max_samples:     Limit the number of volumes (None = all)
        vol_cache_dir:   Dir to cache preprocessed volumes as .npy.
                         MONAI preprocessing (~3-4s/vol) only runs the first time.
                         If None, uses data_dir/lp_vol_cache/.
    """

    def __init__(
        self,
        data_dir: str,
        every_n_epochs: int = 3,
        lp_batch_size: int = 4,
        max_samples: Optional[int] = None,
        vol_cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.data_dir       = Path(data_dir)
        self.every_n_epochs = every_n_epochs
        self.lp_batch_size  = lp_batch_size
        self.max_samples    = max_samples
        self.vol_cache_dir  = Path(vol_cache_dir) if vol_cache_dir else self.data_dir / "lp_vol_cache"
        self.vol_cache_dir.mkdir(parents=True, exist_ok=True)
        (self.vol_cache_dir / "masks").mkdir(exist_ok=True)

        label_dir = self.data_dir / "labels"
        if label_dir.exists():
            self.labels = _load_labels(label_dir)
        else:
            self.labels = {}
            log.warning(f"LPEvalCallback: labels dir not found at {label_dir}")

        self._img_tfm  = None   # lazy init (MONAI heavy import)
        self._msk_tfm  = None

    # ------------------------------------------------------------------

    def _load_vol(self, case_id: str, nifti_path: Path) -> torch.Tensor:
        """Loads a preprocessed volume from the .npy cache, or runs MONAI and caches it."""
        import time
        cache_path = self.vol_cache_dir / f"{case_id}.npy"
        if cache_path.exists():
            print(f"[_load_vol] cache hit: {case_id}", flush=True)
            return torch.from_numpy(np.load(cache_path))   # (1, D, H, W) float32
        print(f"[_load_vol] MONAI start: {case_id}", flush=True)
        if self._img_tfm is None:
            self._img_tfm = _build_transforms()
        t0 = time.time()
        data = self._img_tfm({"image": str(nifti_path)})
        vol = data["image"].float()                         # (1, D, H, W)
        print(f"[_load_vol] MONAI done: {case_id} shape={vol.shape} ({time.time()-t0:.1f}s), saving npy...", flush=True)
        np.save(cache_path, vol.numpy())
        print(f"[_load_vol] npy saved: {cache_path}", flush=True)
        return vol

    def _load_roi_mask(
        self,
        case_id: str,
        task: str,
        mask_nifti: Path,
        vol_shape: tuple,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        Returns roi_mask (1, N_tokens) bool on device, or None if there is no mask.
        Caches the downsampled mask as .npy to avoid re-running MONAI.
        """
        if not mask_nifti.exists():
            return None
        cache_path = self.vol_cache_dir / "masks" / f"{task}_{case_id}.npy"
        if cache_path.exists():
            roi_flat = torch.from_numpy(np.load(cache_path))
        else:
            if self._msk_tfm is None:
                self._msk_tfm = _build_mask_transforms()
            mdata       = self._msk_tfm({"mask": str(mask_nifti)})
            # Move to GPU so avg_pool3d is fast
            mask_tensor = mdata["mask"].unsqueeze(0).to(device)
            if mask_tensor.shape[2:] != vol_shape:
                mask_tensor = F.interpolate(mask_tensor, size=vol_shape, mode="nearest")
            m_down   = F.avg_pool3d(mask_tensor.float(), kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
            roi_flat = (m_down.squeeze() > 0).reshape(-1).cpu()
            np.save(cache_path, roi_flat.numpy())
        if not roi_flat.any():
            return None
        return roi_flat.unsqueeze(0).to(device)            # (1, N_tokens)

    # ------------------------------------------------------------------

    @torch.no_grad()
    def _extract_features(
        self,
        encoder,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
    ) -> Dict[str, np.ndarray]:
        """
        Extracts features for all cases that appear in the CSVs.
        Each rank processes rank::world_size cases to parallelize.
        Returns {case_id: feat_array (out_dim,)} for global and per ROI task.
        """
        img_dir   = self.data_dir / "images"
        mask_root = self.data_dir / "fg_masks"

        all_cases: set = set()
        for task_dict in self.labels.values():
            all_cases.update(task_dict.keys())

        global_feats: Dict[str, np.ndarray] = {}
        roi_feats: Dict[str, Dict[str, np.ndarray]] = {t: {} for t in ROI_TASKS}

        import time as _time
        eager_encoder = getattr(encoder, '_orig_mod', encoder)
        print(f"[LP rank{rank}] eager_encoder type: {type(eager_encoder).__name__}", flush=True)
        eager_encoder.eval()
        cases_list = sorted(all_cases)
        if self.max_samples is not None:
            cases_list = cases_list[: self.max_samples]
        # Each rank processes its slice of the dataset
        local_cases = cases_list[rank::world_size]
        print(f"[LP rank{rank}] processing {len(local_cases)} cases: {local_cases}", flush=True)

        for i, case_id in enumerate(local_cases):
            case_path = img_dir / f"{case_id}.nii.gz"
            if not case_path.exists():
                print(f"[LP rank{rank}] MISSING: {case_path}", flush=True)
                continue

            print(f"[LP rank{rank}] [{i+1}/{len(local_cases)}] start {case_id}", flush=True)
            try:
                t_vol = _time.time()
                vol = self._load_vol(case_id, case_path)
                print(f"[LP rank{rank}] vol loaded {case_id} shape={vol.shape} ({_time.time()-t_vol:.1f}s)", flush=True)
                vol_gpu = vol.unsqueeze(0).to(device)

                print(f"[LP rank{rank}] encoder forward start {case_id}", flush=True)
                t_enc = _time.time()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    tokens = eager_encoder(vol_gpu).squeeze(0)               # (N, D)
                    feat_g = eager_encoder.project(tokens.unsqueeze(0))      # (1, out_dim)
                print(f"[LP rank{rank}] encoder done {case_id} tokens={tokens.shape[0]} ({_time.time()-t_enc:.1f}s)", flush=True)
                global_feats[case_id] = feat_g.squeeze(0).cpu().float().numpy()

                # ROI-masked features
                for task in ROI_TASKS:
                    if case_id not in self.labels.get(task, {}):
                        continue
                    mask_path = mask_root / task / f"{case_id}.nii.gz"
                    roi_mask = self._load_roi_mask(case_id, task, mask_path, vol_gpu.shape[2:], device)
                    if roi_mask is not None:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            feat_r = eager_encoder.project(tokens.unsqueeze(0), roi_mask=roi_mask)
                        roi_feats[task][case_id] = feat_r.squeeze(0).cpu().float().numpy()
                    else:
                        roi_feats[task][case_id] = global_feats[case_id]

                del tokens, feat_g, vol_gpu
                torch.cuda.empty_cache()
                print(f"[LP rank{rank}] done {case_id}", flush=True)

            except Exception as e:
                import traceback
                print(f"[LP rank{rank}] EXCEPTION {case_id}: {e}\n{traceback.format_exc()}", flush=True)
                log.warning(f"LPEvalCallback: error in {case_id}: {e}")
                torch.cuda.empty_cache()

            if (i + 1) % 10 == 0:
                log.info(f"  rank{rank}: {i+1}/{len(local_cases)} volumes processed")

        return global_feats, roi_feats

    # ------------------------------------------------------------------

    def _run_lp(
        self,
        global_feats: Dict[str, np.ndarray],
        roi_feats: Dict[str, Dict[str, np.ndarray]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Runs LogisticRegression per task. Returns {task: {bacc, auroc, vs_curia}}.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score, roc_auc_score
        from sklearn.preprocessing import StandardScaler

        results: Dict[str, Dict[str, float]] = {}

        for task in TASKS:
            if task not in self.labels:
                continue

            feat_dict = roi_feats[task] if task in ROI_TASKS else global_feats
            task_labels = self.labels[task]

            X_tr, y_tr, X_va, y_va = [], [], [], []
            for case_id, (label, split) in task_labels.items():
                if case_id not in feat_dict:
                    continue
                f = feat_dict[case_id]
                if split == "train":
                    X_tr.append(f); y_tr.append(label)
                elif split == "val":
                    X_va.append(f); y_va.append(label)

            if not X_tr or not X_va or len(set(y_va)) < 2:
                continue

            X_tr = np.array(X_tr); y_tr = np.array(y_tr)
            X_va = np.array(X_va); y_va = np.array(y_va)

            scaler = StandardScaler()
            X_tr   = scaler.fit_transform(X_tr)
            X_va   = scaler.transform(X_va)

            try:
                clf = LogisticRegression(C=0.1, max_iter=500, class_weight="balanced", random_state=42)
                clf.fit(X_tr, y_tr)
                y_pred  = clf.predict(X_va)
                y_proba = clf.predict_proba(X_va)[:, 1]
                bacc  = float(balanced_accuracy_score(y_va, y_pred))
                auroc = float(roc_auc_score(y_va, y_proba))
            except Exception as e:
                log.warning(f"LP task {task} failed: {e}")
                continue

            results[task] = {
                "bacc":     bacc,
                "auroc":    auroc,
                "vs_curia": bacc - CURIA_BALACC.get(task, 0.0),
            }

        return results

    # ------------------------------------------------------------------

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = trainer.current_epoch
        rank       = trainer.global_rank
        world_size = trainer.world_size

        # All ranks must agree to exit or continue
        if (epoch + 1) % self.every_n_epochs != 0:
            return
        if not self.labels:
            return

        try:
            import wandb
        except ImportError:
            if rank == 0:
                log.warning("LPEvalCallback: wandb not available, skipping LP eval")
            return

        import torch.distributed as dist
        _ddp = dist.is_available() and dist.is_initialized() and world_size > 1

        # Entry barrier: all ranks must arrive together before starting.
        # Prevents the NCCL watchdog from detecting asymmetric inactivity.
        if _ddp:
            dist.barrier()

        if rank == 0:
            log.info(f"\n[LPEvalCallback] Epoch {epoch+1} - starting LP eval on AMOS val...")
        device = pl_module.device
        encoder = pl_module.encoder

        # Get the uncompiled module first to avoid torch.compile recompilation on .eval()
        eager_encoder = getattr(encoder, '_orig_mod', encoder)
        eager_encoder.eval()
        local_global, local_roi = self._extract_features(encoder, device, rank, world_size)
        eager_encoder.train()

        # Gather results from all ranks
        if _ddp:
            all_global: List[Dict] = [None] * world_size
            all_roi:    List[Dict] = [None] * world_size
            dist.all_gather_object(all_global, local_global)
            dist.all_gather_object(all_roi,    local_roi)
            global_feats: Dict[str, np.ndarray] = {}
            roi_feats: Dict[str, Dict[str, np.ndarray]] = {t: {} for t in ROI_TASKS}
            for g in all_global:
                global_feats.update(g)
            for r in all_roi:
                for t in ROI_TASKS:
                    roi_feats[t].update(r[t])
        else:
            global_feats, roi_feats = local_global, local_roi

        # Non-0 ranks wait here for rank 0 to finish logging (exit barrier).
        # Without this, rank 0 can take minutes in _run_lp while NCCL considers
        # the group blocked and fires a timeout.
        if rank != 0:
            if _ddp:
                dist.barrier()
            return

        results = self._run_lp(global_feats, roi_feats)

        if not results:
            return

        baccs  = [r["bacc"]  for r in results.values()]
        aurocs = [r["auroc"] for r in results.values()]
        mean_bacc  = float(np.mean(baccs))
        mean_auroc = float(np.mean(aurocs))
        mean_vs_curia = float(np.mean([r["vs_curia"] for r in results.values()]))

        log_dict = {
            "val/lp/mean_bacc":     mean_bacc,
            "val/lp/mean_auroc":    mean_auroc,
            "val/lp/vs_curia_mean": mean_vs_curia,
        }
        for task, r in results.items():
            log_dict[f"val/lp/{task}/bacc"]     = r["bacc"]
            log_dict[f"val/lp/{task}/auroc"]    = r["auroc"]
            log_dict[f"val/lp/{task}/vs_curia"] = r["vs_curia"]

        if trainer.logger is not None:
            trainer.logger.experiment.log(log_dict)

        # Console summary
        beats = sum(1 for r in results.values() if r["vs_curia"] > 0)
        log.info(
            f"[LP Epoch {epoch+1}] mean_bacc={mean_bacc:.3f}  mean_auroc={mean_auroc:.3f}"
            f"  vs_curia={mean_vs_curia:+.3f}  beats_curia={beats}/{len(results)}"
        )

        # Exit barrier (rank 0): releases the other ranks waiting at their barrier.
        if _ddp:
            dist.barrier()
