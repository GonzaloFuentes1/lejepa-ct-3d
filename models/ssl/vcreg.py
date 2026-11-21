"""
VCReg: Variance-Covariance Regularizer for SSL collapse prevention.

Adapted from stable-worldmodel/stable_worldmodel/wm/loss.py.
Simplified for 2D (N, D) token embeddings.

Reference: Bardes et al. "VICReg" (arXiv 2105.04906)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VCReg(nn.Module):
    """
    Variance-Covariance Regularizer.

    Enforces two properties on embeddings z ∈ ℝ^(N×D):
      1. Variance ≥ 1 per feature dimension (prevents dimensional collapse)
      2. Near-zero off-diagonal covariance (prevents feature correlation)

    Use alongside SIGReg: VCReg catches dimensional collapse early (direct
    variance penalty), SIGReg enforces the full isotropic Gaussian shape.

    Args:
        eps: Numerical stability constant for std computation.
    """

    def __init__(self, eps: float = 1e-4) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, z: torch.Tensor) -> dict:
        """
        Args:
            z: (N, D) flattened token embeddings (e.g., all_emb.flatten(0, 1)).
        Returns:
            dict with 'std_loss' and 'cov_loss' scalars (both float32).
        """
        z = z.float()
        z = z - z.mean(dim=0, keepdim=True)   # center per feature
        N, D = z.shape

        # Variance loss: penalize dimensions with std < 1
        std = (z.var(dim=0) + self.eps).sqrt()  # (D,)
        std_loss = F.relu(1.0 - std).mean()

        # Covariance loss: penalize off-diagonal correlations
        cov = (z.T @ z) / (N - 1)              # (D, D)
        diag_sq = cov.diagonal().pow(2).sum()
        # Normalized by D*(D-1) so scale is independent of embed_dim
        cov_loss = (cov.pow(2).sum() - diag_sq) / (D * (D - 1) + 1e-9)

        return {"std_loss": std_loss, "cov_loss": cov_loss}
