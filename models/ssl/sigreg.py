"""
SIGReg: Sketched Isotropic Gaussian Regularization wrapper.

Wraps the official EppsPulley test from galilai-group/lejepa with random
projections and a DDP-aware forward pass. If the lejepa package is not
installed, falls back to a self-contained implementation adapted from
le-wm/module.py with added DDP support.

References:
    LeJEPA paper (arXiv 2511.08544), Algorithm 1
    galilai-group/lejepa: lejepa/univariate/epps_pulley.py
    lucas-maes/le-wm: module.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist


try:
    from lejepa.univariate import EppsPulley as _OfficialEppsPulley
    _LEJEPA_AVAILABLE = True
except ImportError:
    _LEJEPA_AVAILABLE = False


class _EppsPulleyFallback(nn.Module):
    """
    Self-contained Epps-Pulley normality test with DDP support.
    Adapted from le-wm/module.py + galilai-group/lejepa/epps_pulley.py.

    Expects pre-projected embeddings x: (N_samples, N_proj).
    Returns a (N_proj,) statistic — call .mean() for scalar loss.
    """

    def __init__(self, t_max: float = 3.0, n_points: int = 17) -> None:
        super().__init__()
        assert n_points % 2 == 1, "n_points must be odd for symmetric quadrature"
        t = torch.linspace(0, t_max, n_points, dtype=torch.float32)
        dt = t_max / (n_points - 1)
        weights = torch.full((n_points,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        phi = t.square().mul(0.5).neg().exp()  # Gaussian CF at t
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)  # Gaussian-windowed trapz

    @property
    def world_size(self) -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N_samples, N_proj) projected embeddings — float32.
        Returns:
            statistic: (N_proj,) per-projection test statistic.
        """
        N = x.size(0)
        # x: (N, P), t: (T,) → x_t: (N, P, T)
        x_t = x.unsqueeze(-1) * self.t

        cos_mean = x_t.cos().mean(0)  # (P, T)
        sin_mean = x_t.sin().mean(0)  # (P, T)

        # DDP: average ECF across all ranks before computing error
        if dist.is_available() and dist.is_initialized():
            # Use functional all_reduce so gradients flow correctly through DDP
            cos_mean = _functional_all_reduce_avg(cos_mean)
            sin_mean = _functional_all_reduce_avg(sin_mean)

        err = (cos_mean - self.phi).square() + sin_mean.square()  # (P, T)
        return (err @ self.weights) * N * self.world_size  # (P,)


def _functional_all_reduce_avg(x: torch.Tensor) -> torch.Tensor:
    """Average a tensor across all DDP ranks, with gradient support."""
    try:
        from torch.distributed.nn import all_reduce as fn_all_reduce
        from torch.distributed.nn import ReduceOp
        return fn_all_reduce(x, ReduceOp.AVG)
    except ImportError:
        # Fallback: in-place (no gradient through communication)
        dist.all_reduce(x, op=dist.ReduceOp.AVG)
        return x


class SIGReg(nn.Module):
    """
    Sketched Isotropic Gaussian Regularization (SIGReg).

    Projects embeddings to 1-D slices and applies the Epps-Pulley
    normality test to enforce isotropic Gaussian embeddings, preventing
    representational collapse without EMA or stop-gradient.

    Uses the official lejepa package when available, otherwise falls back
    to a self-contained implementation.

    Args:
        embed_dim: Dimensionality of input embeddings.
        num_proj: Number of random 1-D projections (slices). Paper uses 256;
            le-wm uses 1024. More projections → better coverage but higher cost.
        t_max: Upper bound of integration interval. Paper: 3.0.
        n_points: Number of quadrature points (must be odd). Paper: 17.

    Example:
        >>> sigreg = SIGReg(embed_dim=384, num_proj=256)
        >>> z = torch.randn(64, 384)   # (B, D) or (B*N_tokens, D)
        >>> loss = sigreg(z)           # scalar
    """

    def __init__(
        self,
        embed_dim: int,
        num_proj: int = 256,
        t_max: float = 3.0,
        n_points: int = 17,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_proj = num_proj

        if _LEJEPA_AVAILABLE:
            self.ep_test = _OfficialEppsPulley(t_max=t_max, n_points=n_points)
        else:
            self.ep_test = _EppsPulleyFallback(t_max=t_max, n_points=n_points)

        # Fixed random projection matrix (not a learned parameter)
        proj = torch.randn(embed_dim, num_proj)
        proj = F.normalize(proj, p=2, dim=0)
        self.register_buffer("proj", proj)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute SIGReg loss over a batch of embeddings.

        Args:
            z: (..., D) embeddings, any leading batch dimensions.
               Typically (B, D) or (B*N_tokens, D).
        Returns:
            Scalar loss value (float32).
        """
        # Always run in float32 — complex exponentials collapse in bf16
        z_f32 = z.detach().float() if not z.requires_grad else z.float()
        z_flat = z_f32.reshape(-1, self.embed_dim)  # (N, D)

        A = self.proj.float()  # (D, num_proj)
        projected = z_flat @ A  # (N, num_proj)

        stat = self.ep_test(projected)  # (num_proj,)
        return stat.mean()
