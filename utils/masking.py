"""
3D Block Masking for volumetric JEPA pretraining.

Implements block masking over a 3D token grid (e.g. 8×8×8 from a 128³
volume with patch_size=16). Block masking forces the predictor to fill
contiguous volumetric regions, which is harder and more useful than
uniform random masking for learning semantic 3D representations.
"""

from __future__ import annotations

import math
import torch
from typing import Tuple


def sample_block_masks_3d(
    grid_shape: Tuple[int, int, int],
    mask_ratio: float = 0.75,
    block_scale_range: Tuple[float, float] = (0.1, 0.5),
    aspect_ratio_range: Tuple[float, float] = (0.3, 3.0),
    n_target_blocks: int = 4,
    seed: int | None = None,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample context and target token indices for 3D block masking.

    Samples ``n_target_blocks`` 3D rectangular blocks as target regions,
    then uses the remaining tokens as context. Follows the strategy from
    I-JEPA extended to 3D.

    Args:
        grid_shape: (Gd, Gh, Gw) number of tokens along each axis.
        mask_ratio: Fraction of tokens to mask (targets). Typical: 0.75.
        block_scale_range: (min, max) fraction of total tokens per block.
        aspect_ratio_range: (min, max) aspect ratio d/h and h/w per block.
        n_target_blocks: Number of separate target blocks to sample.
        seed: Optional RNG seed for reproducibility across DDP ranks.
            Use global_step as seed to get different masks each step while
            keeping all GPUs identical.
        device: Target device for returned tensors.

    Returns:
        ctx_ids: (N_ctx,) flat token indices for context (unmasked).
        tgt_ids: (N_tgt,) flat token indices for targets (masked).
    """
    Gd, Gh, Gw = grid_shape
    N_total = Gd * Gh * Gw

    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    target_mask = torch.zeros(N_total, dtype=torch.bool)

    for _ in range(n_target_blocks):
        scale = _uniform(block_scale_range[0], block_scale_range[1], rng)
        n_block = max(1, int(scale * N_total))

        # Sample block dimensions with aspect ratio constraints
        aspect = _uniform(aspect_ratio_range[0], aspect_ratio_range[1], rng)
        bd = max(1, round((n_block * aspect) ** (1 / 3)))
        bh = max(1, round(bd / aspect))
        bw = max(1, round(n_block / (bd * bh)))

        # Clamp to grid
        bd = min(bd, Gd)
        bh = min(bh, Gh)
        bw = min(bw, Gw)

        # Sample top-left corner
        d0 = int(torch.randint(0, max(1, Gd - bd + 1), (1,), generator=rng).item())
        h0 = int(torch.randint(0, max(1, Gh - bh + 1), (1,), generator=rng).item())
        w0 = int(torch.randint(0, max(1, Gw - bw + 1), (1,), generator=rng).item())

        # Mark block tokens as targets
        d_idx = torch.arange(d0, min(d0 + bd, Gd))
        h_idx = torch.arange(h0, min(h0 + bh, Gh))
        w_idx = torch.arange(w0, min(w0 + bw, Gw))
        dd, hh, ww = torch.meshgrid(d_idx, h_idx, w_idx, indexing="ij")
        flat_ids = dd.reshape(-1) * (Gh * Gw) + hh.reshape(-1) * Gw + ww.reshape(-1)
        target_mask[flat_ids] = True

    tgt_ids = torch.where(target_mask)[0]
    ctx_ids = torch.where(~target_mask)[0]

    # Enforce fixed token budgets so every step has the same sequence lengths.
    # Variable lengths cause torch.compile to recompile and prevent CUDAGraph reuse.
    n_target_budget = int(mask_ratio * N_total)
    n_context_budget = N_total - n_target_budget

    if tgt_ids.numel() > n_target_budget:
        perm = torch.randperm(tgt_ids.numel(), generator=rng)
        tgt_ids = tgt_ids[perm[:n_target_budget]]
        keep_mask = torch.ones(N_total, dtype=torch.bool)
        keep_mask[tgt_ids] = False
        ctx_ids = torch.where(keep_mask)[0]
    elif tgt_ids.numel() < n_target_budget:
        # Pad by borrowing from context (least-recently sampled, i.e. end of ctx_ids)
        n_borrow = n_target_budget - tgt_ids.numel()
        n_borrow = min(n_borrow, ctx_ids.numel() - 1)  # keep at least 1 context token
        tgt_ids = torch.cat([tgt_ids, ctx_ids[-n_borrow:]])
        ctx_ids = ctx_ids[:-n_borrow]

    if ctx_ids.numel() > n_context_budget:
        ctx_ids = ctx_ids[:n_context_budget]

    if device is not None:
        ctx_ids = ctx_ids.to(device)
        tgt_ids = tgt_ids.to(device)

    return ctx_ids, tgt_ids


def _uniform(lo: float, hi: float, rng: torch.Generator) -> float:
    return lo + (hi - lo) * torch.rand(1, generator=rng).item()


def make_3d_sinusoidal_pos_embed(
    embed_dim: int,
    grid_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """
    Build 3D sinusoidal positional embeddings for a token grid.

    Args:
        embed_dim: Embedding dimension (must be divisible by 6).
        grid_shape: (Gd, Gh, Gw).

    Returns:
        pos_embed: (Gd*Gh*Gw, embed_dim) positional embeddings.
    """
    Gd, Gh, Gw = grid_shape
    assert embed_dim % 6 == 0, "embed_dim must be divisible by 6 for 3D sinusoidal PE"
    dim_per_axis = embed_dim // 3

    def sinusoidal_1d(length: int, dim: int) -> torch.Tensor:
        position = torch.arange(length).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(length, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # (length, dim)

    pe_d = sinusoidal_1d(Gd, dim_per_axis)  # (Gd, D/3)
    pe_h = sinusoidal_1d(Gh, dim_per_axis)  # (Gh, D/3)
    pe_w = sinusoidal_1d(Gw, dim_per_axis)  # (Gw, D/3)

    # Broadcast and concatenate
    pe_d = pe_d[:, None, None, :].expand(Gd, Gh, Gw, -1)
    pe_h = pe_h[None, :, None, :].expand(Gd, Gh, Gw, -1)
    pe_w = pe_w[None, None, :, :].expand(Gd, Gh, Gw, -1)

    pos_embed = torch.cat([pe_d, pe_h, pe_w], dim=-1)  # (Gd, Gh, Gw, D)
    return pos_embed.reshape(Gd * Gh * Gw, embed_dim)
