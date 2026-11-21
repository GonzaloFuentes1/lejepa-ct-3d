"""
Flow Matching networks for lejepa_diff_local and lejepa_diff_global modes.

Replaces DDPM (cosine schedule, T=1000 integer steps) with Conditional Flow Matching:
    x_t = (1-t)*x_0 + t*x_1,   t ~ Uniform(0,1)
    velocity = x_1 - x_0        (constant along linear path)
    loss = MSE(v_θ(x_t, t, cond), velocity)

References:
    - Lipman et al. "Flow Matching for Generative Modeling" (2022)
    - Liu et al. "Flow Straight and Fast: Rectified Flow" (2022)
    - Albergo & Vanden-Eijnden "Building Normalizing Flows with Stochastic Interpolants" (2022)

Advantages over DDPM for SSL pretraining:
    - Simpler: no schedule hyperparameters (T, β_min, β_max, s)
    - Straighter trajectories → faster convergence
    - t is continuous in [0,1] — no discretization artifacts
    - At inference (optional): Euler solver in K steps (K=1 often sufficient)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# AdaLN-zero Transformer block (from le-wm/module.py, unchanged)
# ---------------------------------------------------------------------------

class _FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConditionalBlock(nn.Module):
    """
    Transformer block with AdaLN-zero conditioning (DiT-style).
    Conditioning vector c modulates activations via learned scale/shift/gate.
    Zero-initialized final projection ensures identity at init.
    """

    def __init__(self, dim: int, num_heads: int = 6, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = _FeedForward(dim, int(dim * mlp_ratio), dropout=dropout)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(c).chunk(6, dim=-1)
        shift_msa, scale_msa, gate_msa = [v.unsqueeze(1) for v in (shift_msa, scale_msa, gate_msa)]
        shift_mlp, scale_mlp, gate_mlp = [v.unsqueeze(1) for v in (shift_mlp, scale_mlp, gate_mlp)]
        x_n = self.norm1(x) * (1 + scale_msa) + shift_msa
        attn_out, _ = self.attn(x_n, x_n, x_n)
        x = x + gate_msa * attn_out
        x_n = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.mlp(x_n)
        return x


# ---------------------------------------------------------------------------
# Time embedding — works for both continuous t ∈ [0,1] and integer t
# ---------------------------------------------------------------------------

class TimestepEmbedding(nn.Module):
    """Sinusoidal embedding for continuous t ∈ [0,1], projected to hidden_dim."""

    def __init__(self, hidden_dim: int, freq_dim: int = 256) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        half = freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half) / half)
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) float in [0,1]. Returns (B, hidden_dim)."""
        t_f = t.float().unsqueeze(1) * self.freqs.unsqueeze(0)  # (B, half)
        emb = torch.cat([t_f.cos(), t_f.sin()], dim=-1)          # (B, freq_dim)
        return self.proj(emb)


# ---------------------------------------------------------------------------
# Flow Matching schedule
# ---------------------------------------------------------------------------

class FlowMatchingSchedule:
    """
    Conditional Flow Matching with linear (rectified) paths.

    Forward path:  x_t = (1-t)*x_0 + t*x_1
    Velocity:      v   = x_1 - x_0   (constant along each path)
    Training loss: MSE(v_θ(x_t, t, cond), v)

    x_0 ~ N(0, I)  — source (noise)
    x_1             — target (data embedding, detached)
    t  ~ U(0, 1)   — continuous, no discretization
    """

    @staticmethod
    def sample_t(B: int, device: torch.device) -> torch.Tensor:
        """Sample t ~ Uniform(0, 1), shape (B,)."""
        return torch.rand(B, device=device)

    @staticmethod
    def interpolate(
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute interpolated x_t and constant target velocity.

        Args:
            x0: (B, ...) Gaussian noise.
            x1: (B, ...) target data embedding (detached).
            t:  (B,) timesteps in [0, 1].
        Returns:
            x_t:      Interpolated sample at time t.
            velocity: x_1 - x_0 (the vector field to learn).
        """
        # Expand t to match x1 dims for broadcasting
        t_exp = t
        for _ in range(x1.dim() - 1):
            t_exp = t_exp.unsqueeze(-1)
        x_t = (1.0 - t_exp) * x0 + t_exp * x1
        velocity = x1 - x0
        return x_t, velocity

    @staticmethod
    def euler_step(
        x_t: torch.Tensor,
        v: torch.Tensor,
        t: float,
        dt: float,
    ) -> torch.Tensor:
        """One Euler step along the flow: x_{t+dt} = x_t + dt * v."""
        return x_t + dt * v

    @classmethod
    def sample(
        cls,
        net: nn.Module,
        shape: tuple,
        steps: int = 10,
        ctx_emb: Optional[torch.Tensor] = None,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Generate samples via Euler ODE solver (for inference/visualization only).
        Not used during training.

        Args:
            net:     FlowNetPatch or FlowNetGlobal.
            shape:   Shape of output tensor (e.g. (B, N_tgt, D)).
            steps:   Number of Euler steps (10 is usually enough).
            ctx_emb: Context embeddings for conditional generation.
            device:  Target device.
        Returns:
            x_1: Generated sample starting from x_0 ~ N(0,I).
        """
        x = torch.randn(*shape, device=device)
        dt = 1.0 / steps
        ts = torch.linspace(0.0, 1.0 - dt, steps, device=device)
        for t_val in ts:
            t = torch.full((shape[0],), t_val, device=device)
            with torch.no_grad():
                if ctx_emb is not None:
                    v = net(x, t, ctx_emb)
                else:
                    v = net(x, t)
            x = cls.euler_step(x, v, t_val, dt)
        return x


# ---------------------------------------------------------------------------
# Flow networks (replace ScoreNetPatch / ScoreNetGlobal)
# ---------------------------------------------------------------------------

class FlowNetPatch(nn.Module):
    """
    Velocity field network for lejepa_diff_local.

    Predicts the flow velocity v(x_t, t | ctx) that transports noise → target
    patch embeddings, conditioned on context encoder outputs.

    Architecture: same ConditionalBlock stack as before, but:
    - Output is velocity (not noise ε)
    - t is continuous in [0,1] (not integer steps)
    - No schedule hyperparameters needed

    Args:
        embed_dim: Latent dimension (matches encoder).
        depth:     Number of ConditionalBlock layers.
        num_heads: Attention heads.
    """

    def __init__(self, embed_dim: int = 384, depth: int = 4, num_heads: int = 6) -> None:
        super().__init__()
        self.schedule = FlowMatchingSchedule()
        self.t_embed = TimestepEmbedding(hidden_dim=embed_dim)
        self.blocks = nn.ModuleList([ConditionalBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        ctx_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict velocity v(x_t, t | ctx).

        Args:
            x_t:     (B, N_tgt, D) interpolated patch embeddings at time t.
            t:       (B,) continuous timesteps in [0, 1].
            ctx_emb: (B, N_ctx, D) context encoder outputs.
        Returns:
            v_pred: (B, N_tgt, D) predicted velocity.
        """
        c = self.t_embed(t) + ctx_emb.mean(1)  # (B, D)
        x = x_t
        for block in self.blocks:
            x = block(x, c)
        return self.out_proj(self.norm(x))


class FlowNetGlobal(nn.Module):
    """
    Velocity field network for lejepa_diff_global.

    Unconditional: transports noise → global volume embedding (mean-pooled tokens).
    No context conditioning — the model learns the marginal embedding distribution.

    Args:
        embed_dim: Latent dimension.
        depth:     Number of ConditionalBlock layers.
        num_heads: Attention heads.
    """

    def __init__(self, embed_dim: int = 384, depth: int = 4, num_heads: int = 6) -> None:
        super().__init__()
        self.schedule = FlowMatchingSchedule()
        self.t_embed = TimestepEmbedding(hidden_dim=embed_dim)
        self.blocks = nn.ModuleList([ConditionalBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: (B, D) global embedding at time t.
            t:   (B,) continuous timesteps in [0, 1].
        Returns:
            v_pred: (B, D) predicted velocity.
        """
        c = self.t_embed(t)      # (B, D)
        x = x_t.unsqueeze(1)    # (B, 1, D)
        for block in self.blocks:
            x = block(x, c)
        return self.out_proj(self.norm(x)).squeeze(1)  # (B, D)

