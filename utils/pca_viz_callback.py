"""
PCA Feature Visualization Callback for W&B.

Every `every_n_steps` steps:
  1. Takes a fixed batch from val_loader (always the same volume → comparable evolution)
  2. Extracts all encoder tokens (B=1, N=600, D=1024)
  3. PCA to 3 components → normalized RGB [0,1]
  4. Reshapes to the spatial grid (Gd, Gh, Gw) and visualizes the 3 orthogonal views:
       Axial   (z=middle): slice in the H×W plane
       Coronal (y=middle): slice in the D×W plane
       Sagittal (x=middle): slice in the D×H plane
  5. Logs to W&B as an image with the CT volume as a grayscale background

Inspired by Figure 1 of the LeJEPA paper (Balestriero & LeCun, 2025).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback


class PCAVizCallback(Callback):
    """
    Callback that logs PCA visualizations of the encoder embeddings to W&B.

    Args:
        val_loader:     Validation DataLoader (the first batch is fixed).
        every_n_steps:  Logging frequency in training steps.
        patch_size:     Encoder patch size (default 16).
        n_components:   PCA components (3 → RGB).
        max_vols:       How many volumes of the batch to visualize (default 1).
    """

    def __init__(
        self,
        val_loader,
        every_n_steps: int = 1000,
        patch_size: int = 16,
        n_components: int = 3,
        max_vols: int = 1,
        pool_size: int = 5,
    ) -> None:
        super().__init__()
        self.every_n_steps = every_n_steps
        self.patch_size     = patch_size
        self.n_components   = n_components
        self.max_vols       = max_vols
        self._pool_size     = pool_size
        self._batch_pool: list = []
        self._val_loader    = val_loader
        self._val_iter      = None   # lazy init (loader not iterable at __init__ time)

    # ------------------------------------------------------------------

    def _get_batch(self, step: int, device: torch.device) -> torch.Tensor:
        """Returns a volume from the rotating pool (different at each trigger)."""
        if self._val_iter is None:
            self._val_iter = iter(self._val_loader)

        # Index that rotates across the pool_size slots
        slot = (step // self.every_n_steps) % self._pool_size

        # Fill the pool up to the needed slot
        while len(self._batch_pool) <= slot:
            try:
                batch = next(self._val_iter)
            except StopIteration:
                self._val_iter = iter(self._val_loader)
                batch = next(self._val_iter)
            x = batch["image"] if isinstance(batch, dict) else batch
            self._batch_pool.append(x[: self.max_vols].cpu())

        return self._batch_pool[slot].to(device)

    # ------------------------------------------------------------------

    @torch.no_grad()
    def _pca_rgb(self, tokens: torch.Tensor) -> np.ndarray:
        """
        tokens: (N, D) float32
        Returns: (N, 3) float32 in [0,1] with the first 3 PCA components normalized.
        """
        z = tokens.float()
        z = z - z.mean(0)

        # SVD for PCA (more stable than eig in bf16)
        try:
            _, _, Vh = torch.linalg.svd(z, full_matrices=False)
            components = Vh[: self.n_components]        # (3, D)
            proj = z @ components.T                     # (N, 3)
        except Exception:
            proj = z[:, : self.n_components]

        # Normalize each component to [0, 1]
        proj = proj - proj.min(0).values
        denom = proj.max(0).values - proj.min(0).values + 1e-8
        proj = proj / denom
        return proj.cpu().numpy().astype(np.float32)    # (N, 3)

    # ------------------------------------------------------------------

    @torch.no_grad()
    def _make_panel(
        self,
        vol: torch.Tensor,      # (1, D, H, W) float32 normalized [0,1]
        tokens: torch.Tensor,   # (N, embed_dim)
        grid: tuple,            # (Gd, Gh, Gw)
    ) -> np.ndarray:
        """
        Builds a panel with 3 orthogonal views:
          row 1: CT in grayscale
          row 2: PCA features (colored)
        Returns: (H_panel, W_panel, 3) uint8
        """
        Gd, Gh, Gw = grid
        pca_rgb = self._pca_rgb(tokens)          # (N, 3)
        pca_vol = pca_rgb.reshape(Gd, Gh, Gw, 3) # (Gd, Gh, Gw, 3)

        P = self.patch_size
        D, H, W = vol.shape[1], vol.shape[2], vol.shape[3]

        # Upsample the PCA grid to the full volume resolution
        pca_t = torch.from_numpy(pca_vol).permute(3, 0, 1, 2).unsqueeze(0)  # (1,3,Gd,Gh,Gw)
        pca_full = F.interpolate(pca_t, size=(D, H, W), mode="trilinear", align_corners=False)  # (1,3,D,H,W)
        pca_np = pca_full.squeeze(0).permute(1, 2, 3, 0).numpy()             # (D,H,W,3)

        ct_np = vol.squeeze(0).numpy()  # (D,H,W) in [0,1]

        # Middle slices in each plane
        zd = D // 2
        zh = H // 2
        zw = W // 2

        def to_rgb(gray: np.ndarray) -> np.ndarray:
            """Gray [0,1] → RGB uint8."""
            return (np.stack([gray, gray, gray], -1) * 255).astype(np.uint8)

        def to_uint8(rgb: np.ndarray) -> np.ndarray:
            return (rgb * 255).clip(0, 255).astype(np.uint8)

        # Vistas CT
        axial_ct   = to_rgb(ct_np[zd])           # (H, W, 3)
        coronal_ct = to_rgb(ct_np[:, zh, :])      # (D, W, 3)
        sagital_ct = to_rgb(ct_np[:, :, zw])      # (D, H, 3)

        # Vistas PCA
        axial_pca   = to_uint8(pca_np[zd])        # (H, W, 3)
        coronal_pca = to_uint8(pca_np[:, zh, :])  # (D, W, 3)
        sagital_pca = to_uint8(pca_np[:, :, zw])  # (D, H, 3)

        # Equalize heights to concatenate horizontally
        def pad_h(img: np.ndarray, target_h: int) -> np.ndarray:
            if img.shape[0] < target_h:
                pad = np.zeros((target_h - img.shape[0], img.shape[1], 3), dtype=img.dtype)
                img = np.concatenate([img, pad], 0)
            return img

        target_h = max(H, D)
        axial_ct    = pad_h(axial_ct,   target_h)
        coronal_ct  = pad_h(coronal_ct, target_h)
        sagital_ct  = pad_h(sagital_ct, target_h)
        axial_pca   = pad_h(axial_pca,   target_h)
        coronal_pca = pad_h(coronal_pca, target_h)
        sagital_pca = pad_h(sagital_pca, target_h)

        # 2px vertical separator
        sep = np.ones((target_h, 2, 3), dtype=np.uint8) * 128

        row_ct  = np.concatenate([axial_ct,  sep, coronal_ct,  sep, sagital_ct],  axis=1)
        row_pca = np.concatenate([axial_pca, sep, coronal_pca, sep, sagital_pca], axis=1)

        # Horizontal separator
        sep_h = np.ones((4, row_ct.shape[1], 3), dtype=np.uint8) * 128
        panel = np.concatenate([row_ct, sep_h, row_pca], axis=0)

        return panel

    # ------------------------------------------------------------------

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step == 0 or step % self.every_n_steps != 0:
            return
        if trainer.logger is None:
            return

        # Only log from rank 0
        if trainer.global_rank != 0:
            return

        try:
            import wandb
        except ImportError:
            return

        device = pl_module.device
        encoder = pl_module.encoder
        patch_size = self.patch_size

        x = self._get_batch(step, device)           # (B, 1, D, H, W)
        B, C, D, H, W = x.shape
        Gd = D // patch_size
        Gh = H // patch_size
        Gw = W // patch_size
        grid = (Gd, Gh, Gw)

        encoder.eval()
        with torch.no_grad():
            if x.dtype == torch.float16 or next(encoder.parameters()).dtype == torch.float16:
                x_in = x.half()
            else:
                x_in = x.float()

            tokens = encoder(x_in)   # (B, N, embed_dim)

        images = []
        for i in range(min(B, self.max_vols)):
            panel = self._make_panel(
                vol=x[i].float().cpu(),
                tokens=tokens[i].float().cpu(),
                grid=grid,
            )
            images.append(wandb.Image(panel, caption=f"vol_{i} step={step}"))

        trainer.logger.experiment.log(
            {"val/pca_features": images},
        )
        encoder.train()
