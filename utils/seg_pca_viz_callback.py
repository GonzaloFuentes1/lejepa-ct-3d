"""
Segmentation-overlay PCA Visualization Callback.

Cada `every_n_steps` pasos loguea a W&B un panel con 3 filas y 3 columnas
(vistas axial / coronal / sagital):
  Fila 1: CT en escala de grises
  Fila 2: PCA features del encoder (interpolación trilinear → suave)
  Fila 3: Máscara de segmentación coloreada por órgano

Si el encoder aprende bien, los colores PCA deberían alinearse con los
bordes de los órganos sin haber visto ninguna etiqueta durante el pretraining.

Usa volúmenes de abdomen_organ_seg (13 órganos: hígado, riñones, bazo...).
Rota entre `pool_size` volúmenes distintos para ver consistencia.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback

# Colores por órgano (BGR ignorado — usamos RGB uint8)
# 13 órganos + background → paleta distinguible
ORGAN_COLORS = np.array([
    [0,   0,   0  ],  # 0  background — negro
    [255, 80,  80 ],  # 1  liver       — rojo
    [80,  160, 255],  # 2  right-kidney — azul
    [80,  255, 80 ],  # 3  spleen      — verde
    [255, 200, 80 ],  # 4  pancreas    — naranja
    [200, 80,  255],  # 5  aorta       — violeta
    [80,  255, 230],  # 6  ivc         — cyan
    [255, 255, 80 ],  # 7  rag         — amarillo
    [180, 255, 120],  # 8  lag         — verde claro
    [255, 140, 200],  # 9  gallbladder — rosa
    [120, 200, 255],  # 10 esophagus  — azul claro
    [255, 180, 120],  # 11 stomach    — salmón
    [160, 120, 255],  # 12 duodenum   — lila
    [80,  200, 160],  # 13 left-kidney — turquesa
], dtype=np.uint8)

SEG_DIR = Path(os.environ.get("LEJEPA_DATA_ROOT", "./data")) / "flare_ct_pretrain_full" / "val_downstream" / "abdomen_organ_seg"
SPACING    = (1.5, 1.5, 1.5)
HU_MIN     = -1000.0
HU_MAX     = 1000.0


def _build_transforms():
    from monai.transforms import (
        Compose, EnsureChannelFirstd, EnsureTyped,
        LoadImaged, Orientationd, ScaleIntensityRanged, Spacingd,
    )
    keys = ["image", "label"]
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=list(SPACING), mode="bilinear"),
        Spacingd(keys=["label"], pixdim=list(SPACING), mode="nearest"),
        ScaleIntensityRanged(
            keys=["image"], a_min=HU_MIN, a_max=HU_MAX,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        EnsureTyped(keys=["image"], dtype="float32"),
        EnsureTyped(keys=["label"], dtype="int32"),
    ])


class SegPCAVizCallback(Callback):
    """
    Loguea visualización PCA + segmentación a W&B cada `every_n_steps` pasos.

    Args:
        seg_dir:        Directorio abdomen_organ_seg/ (contiene imagesVal/ y labelsVal/)
        every_n_steps:  Frecuencia de logging (default 1000)
        patch_size:     Patch size del encoder (default 16)
        pool_size:      Número de volúmenes distintos a rotar (default 4)
    """

    def __init__(
        self,
        seg_dir: str | Path = SEG_DIR,
        every_n_steps: int = 1000,
        patch_size: int = 16,
        pool_size: int = 4,
    ) -> None:
        super().__init__()
        self.seg_dir       = Path(seg_dir)
        self.every_n_steps = every_n_steps
        self.patch_size    = patch_size
        self.pool_size     = pool_size

        self._tfm = None   # lazy init
        self._pool: list   = []   # lista de (vol_tensor, label_np)
        self._cases: list  = []   # rutas de casos disponibles

        img_dir = self.seg_dir / "imagesVal"
        lbl_dir = self.seg_dir / "labelsVal"
        if img_dir.exists() and lbl_dir.exists():
            imgs = sorted(img_dir.glob("*.nii.gz"))
            for img_path in imgs:
                # buscar label correspondiente (mismo stem sin _0000)
                stem = img_path.name.replace("_0000.nii.gz", "")
                lbl_path = lbl_dir / f"{stem}.nii.gz"
                if lbl_path.exists():
                    self._cases.append((img_path, lbl_path))

    # ------------------------------------------------------------------

    def _load_case(self, idx: int):
        """Carga y preprocesa el caso idx → (vol float32 cpu, label int32 cpu)."""
        if self._tfm is None:
            self._tfm = _build_transforms()

        img_path, lbl_path = self._cases[idx % len(self._cases)]
        data = self._tfm({"image": str(img_path), "label": str(lbl_path)})
        vol   = data["image"].float()   # (1, D, H, W)
        label = data["label"]           # (1, D, H, W) int32
        return vol.cpu(), label.cpu()

    def _get_pool_entry(self, slot: int):
        while len(self._pool) <= slot:
            vol, label = self._load_case(len(self._pool))
            self._pool.append((vol, label))
        return self._pool[slot]

    # ------------------------------------------------------------------

    @torch.no_grad()
    def _pca_rgb(self, tokens: torch.Tensor) -> np.ndarray:
        """tokens: (N, D) → (N, 3) float32 en [0,1]."""
        z = tokens.float()
        z = z - z.mean(0)
        try:
            _, _, Vh = torch.linalg.svd(z, full_matrices=False)
            proj = z @ Vh[:3].T
        except Exception:
            proj = z[:, :3]
        proj = proj - proj.min(0).values
        proj = proj / (proj.max(0).values - proj.min(0).values + 1e-8)
        return proj.cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------

    def _seg_rgb(self, label_np: np.ndarray) -> np.ndarray:
        """
        label_np: (D, H, W) int — convierte a (D, H, W, 3) uint8 con colores por órgano.
        """
        label_clipped = np.clip(label_np, 0, len(ORGAN_COLORS) - 1)
        return ORGAN_COLORS[label_clipped]   # (D, H, W, 3)

    # ------------------------------------------------------------------

    def _make_panel(
        self,
        vol: torch.Tensor,      # (1, D, H, W) float [0,1]
        tokens: torch.Tensor,   # (N, embed_dim)
        label: torch.Tensor,    # (1, D, H, W) int32
        grid: tuple,            # (Gd, Gh, Gw)
    ) -> np.ndarray:
        Gd, Gh, Gw = grid
        P = self.patch_size
        D, H, W = vol.shape[1], vol.shape[2], vol.shape[3]

        # PCA trilinear → suave
        pca_rgb  = self._pca_rgb(tokens)                          # (N, 3)
        pca_vol  = pca_rgb.reshape(Gd, Gh, Gw, 3)
        pca_t    = torch.from_numpy(pca_vol).permute(3, 0, 1, 2).unsqueeze(0)  # (1,3,Gd,Gh,Gw)
        pca_full = F.interpolate(pca_t, size=(D, H, W), mode="trilinear", align_corners=False)
        pca_np   = pca_full.squeeze(0).permute(1, 2, 3, 0).numpy()             # (D,H,W,3)

        ct_np    = vol.squeeze(0).numpy()                         # (D, H, W)

        # Segmentación coloreada (resize al mismo espacio que vol)
        lbl_np   = label.squeeze(0).numpy().astype(np.int32)      # (D, H, W)
        if lbl_np.shape != (D, H, W):
            lbl_t  = torch.from_numpy(lbl_np).unsqueeze(0).unsqueeze(0).float()
            lbl_t  = F.interpolate(lbl_t, size=(D, H, W), mode="nearest")
            lbl_np = lbl_t.squeeze().numpy().astype(np.int32)
        seg_np = self._seg_rgb(lbl_np)                            # (D, H, W, 3)

        zd, zh, zw = D // 2, H // 2, W // 2

        def to_rgb(gray: np.ndarray) -> np.ndarray:
            return (np.stack([gray, gray, gray], -1) * 255).astype(np.uint8)

        def to_uint8(rgb: np.ndarray) -> np.ndarray:
            return (rgb * 255).clip(0, 255).astype(np.uint8)

        # Fila CT
        axial_ct   = to_rgb(ct_np[zd])
        coronal_ct = to_rgb(ct_np[:, zh, :])
        sagital_ct = to_rgb(ct_np[:, :, zw])

        # Fila PCA
        axial_pca   = to_uint8(pca_np[zd])
        coronal_pca = to_uint8(pca_np[:, zh, :])
        sagital_pca = to_uint8(pca_np[:, :, zw])

        # Fila Seg
        axial_seg   = seg_np[zd]
        coronal_seg = seg_np[:, zh, :]
        sagital_seg = seg_np[:, :, zw]

        target_h = max(H, D)

        def pad_h(img, h):
            if img.shape[0] < h:
                img = np.concatenate(
                    [img, np.zeros((h - img.shape[0], img.shape[1], 3), dtype=img.dtype)], 0
                )
            return img

        rows = []
        sep_v = np.ones((target_h, 2, 3), dtype=np.uint8) * 100
        for a, c, s in [
            (axial_ct,  coronal_ct,  sagital_ct),
            (axial_pca, coronal_pca, sagital_pca),
            (axial_seg, coronal_seg, sagital_seg),
        ]:
            a, c, s = pad_h(a, target_h), pad_h(c, target_h), pad_h(s, target_h)
            rows.append(np.concatenate([a, sep_v, c, sep_v, s], axis=1))

        sep_h = np.ones((3, rows[0].shape[1], 3), dtype=np.uint8) * 100
        panel = np.concatenate([rows[0], sep_h, rows[1], sep_h, rows[2]], axis=0)
        return panel

    # ------------------------------------------------------------------

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step == 0 or step % self.every_n_steps != 0:
            return
        if trainer.global_rank != 0:
            return
        if not self._cases:
            return
        if trainer.logger is None:
            return

        try:
            import wandb
        except ImportError:
            return

        slot = (step // self.every_n_steps) % self.pool_size
        vol, label = self._get_pool_entry(slot)

        device = pl_module.device
        encoder = pl_module.encoder
        P = self.patch_size
        x = vol.unsqueeze(0).to(device)   # (1,1,D,H,W)
        D, H, W = x.shape[2], x.shape[3], x.shape[4]
        grid = (D // P, H // P, W // P)

        encoder.eval()
        with torch.no_grad():
            tokens = encoder(x.half() if next(encoder.parameters()).dtype == torch.float16
                             else x.float())   # (1, N, embed_dim)

        panel = self._make_panel(
            vol=vol,
            tokens=tokens[0].float().cpu(),
            label=label,
            grid=grid,
        )

        case_name = self._cases[slot % len(self._cases)][0].stem.replace("_0000", "")
        trainer.logger.experiment.log(
            {"val/pca_seg": [wandb.Image(panel, caption=f"{case_name} step={step}")]},
        )
        encoder.train()
