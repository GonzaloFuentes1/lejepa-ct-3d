"""
ViT-Small-3D: Vision Transformer backbone for 3D volumetric data.

Adapts the standard 2D ViT to 3D patch tokenization. The encoder supports
partial forward passes (given a subset of token indices) for efficient
JEPA masking — only the requested tokens are processed.

Architecture (ViT-Small defaults):
    embed_dim = 384, depth = 12, num_heads = 6, mlp_ratio = 4
    ~22M parameters for 512 tokens (128³ volume, patch_size=16)
"""

from __future__ import annotations

import math
from typing import List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as grad_ckpt
from einops import rearrange

from utils.masking import make_3d_sinusoidal_pos_embed


class PatchEmbed3D(nn.Module):
    """
    3D patch embedding: volume → sequence of tokens.

    Args:
        in_channels: Number of input image channels (1 for CT).
        patch_size: Cubic patch size. Typical: 16.
        embed_dim: Token embedding dimension.
    """

    def __init__(
        self,
        in_channels: int = 1,
        patch_size: int = 16,
        embed_dim: int = 384,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """
        Args:
            x: (B, C, D, H, W) input volume.
        Returns:
            tokens: (B, N, D) patch tokens.
            grid_shape: (Gd, Gh, Gw) token grid dimensions.
        """
        B, C, D, H, W = x.shape
        tokens = self.proj(x)  # (B, E, Gd, Gh, Gw)
        Gd, Gh, Gw = tokens.shape[2:]
        tokens = rearrange(tokens, "b e d h w -> b (d h w) e")
        tokens = self.norm(tokens)
        return tokens, (Gd, Gh, Gw)


class Attention3D(nn.Module):
    """Multi-head self-attention with optional causal masking."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        drop = self.attn_drop if self.training else 0.0
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=drop)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


class Block3D(nn.Module):
    """Standard ViT transformer block with pre-norm."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention3D(embed_dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT3D(nn.Module):
    """
    ViT-Small-3D encoder for volumetric CT data.

    Supports partial forward (passing only a subset of token indices) for
    efficient JEPA masking — no wasted computation on masked tokens.

    Args:
        in_channels: Input channels (1 for single-modality CT).
        patch_size: Cubic patch size. Typical: 16.
        embed_dim: Token embedding dimension. ViT-S: 384.
        depth: Number of transformer blocks. ViT-S: 12.
        num_heads: Attention heads. ViT-S: 6.
        mlp_ratio: MLP hidden / embed ratio. Standard: 4.0.
        drop_rate: Dropout on MLP and projection layers.
        attn_drop_rate: Dropout on attention weights.
        volume_size: Expected input volume (D, H, W) for PE precomputation.
            If None, PE is computed on first forward pass and cached.
        grad_checkpointing: If True, use gradient checkpointing on transformer
            blocks. Reduces VRAM ~40% at the cost of ~20% extra compute.
            Recommended for ViT-H or when batch_size_per_gpu > 16.
    """

    def __init__(
        self,
        in_channels: int = 1,
        patch_size: int = 16,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        volume_size: Optional[Tuple[int, int, int]] = (128, 128, 128),
        grad_checkpointing: bool = False,
        proj_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.grad_checkpointing = grad_checkpointing

        self.patch_embed = PatchEmbed3D(in_channels, patch_size, embed_dim)
        self.drop = nn.Dropout(drop_rate)

        self.blocks = nn.ModuleList([
            Block3D(embed_dim, num_heads, mlp_ratio, drop_rate, attn_drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Output projection: embed_dim → proj_dim (e.g. 1024 → 2048).
        # Applied per-token inside forward(), so encoder always outputs out_dim tokens.
        # Predictor and SIGReg both operate in out_dim space.
        self.proj_head: Optional[nn.Module] = (
            nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, proj_dim))
            if proj_dim is not None else None
        )
        self.out_dim: int = proj_dim if proj_dim is not None else embed_dim

        # Precompute PE if volume_size is known
        if volume_size is not None:
            Gd = volume_size[0] // patch_size
            Gh = volume_size[1] // patch_size
            Gw = volume_size[2] // patch_size
            # embed_dim must be divisible by 6 for 3D sinusoidal PE
            if embed_dim % 6 == 0:
                pe = make_3d_sinusoidal_pos_embed(embed_dim, (Gd, Gh, Gw))
            else:
                # Learned PE fallback when embed_dim % 6 != 0
                N = Gd * Gh * Gw
                pe = torch.zeros(N, embed_dim)
            self.register_buffer("pos_embed", pe)  # (N, D)
        else:
            self.pos_embed = None

        self._init_weights()

    def project(
        self,
        tokens: torch.Tensor,
        roi_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pool tokens → global feature vector.
        Projection is already applied per-token in forward(), so this just pools.

        Args:
            tokens:   (B, N, out_dim) encoder output (already projected).
            roi_mask: (B, N) bool mask. If None, global mean pool.
        Returns:
            (B, out_dim) feature vector.
        """
        if roi_mask is not None:
            mask_f = roi_mask.float().unsqueeze(-1)          # (B, N, 1)
            pooled = (tokens * mask_f).sum(dim=1) / (mask_f.sum(dim=1) + 1e-8)
        else:
            pooled = tokens.mean(dim=1)                       # (B, out_dim)
        return pooled

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _get_pos_embed(
        self, grid_shape: Tuple[int, int, int], device: torch.device
    ) -> torch.Tensor:
        """Return positional embeddings for grid_shape, computing on-the-fly if needed."""
        N = math.prod(grid_shape)
        # Fast path: cached embed matches current grid (training)
        if self.pos_embed is not None and self.pos_embed.shape[0] == N:
            return self.pos_embed.to(device)
        # Compute on-the-fly (first training call or eval with different volume size)
        embed_dim = self.embed_dim
        pe = (make_3d_sinusoidal_pos_embed(embed_dim, grid_shape)
              if embed_dim % 6 == 0
              else torch.zeros(N, embed_dim)).to(device)
        if self.pos_embed is None:
            self.register_buffer("pos_embed", pe)
        return pe

    def forward(
        self,
        x: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        return_layers: Optional[Set[int]] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Encode a volume, optionally restricting to a subset of tokens.

        Args:
            x: (B, C, D, H, W) input volume.
            token_ids: (N_subset,) flat token indices to process.
                If None, all tokens are processed.
            return_layers: Set of block indices whose normed outputs to return.
                E.g. {8, 16, 20, 23} for multi-layer feature extraction.
                When set, returns a list of (B, N, D) tensors (one per requested
                layer, in ascending order) plus the final output appended last.
                When None (default), returns a single (B, N, D) tensor.
        Returns:
            embeddings: (B, N_subset, embed_dim) if return_layers is None.
            list of (B, N_subset, embed_dim) if return_layers is not None,
                ordered by layer index ascending.
        """
        tokens, grid_shape = self.patch_embed(x)  # (B, N, D)
        pe = self._get_pos_embed(grid_shape, x.device)  # (N, D)

        if token_ids is not None:
            tokens = tokens[:, token_ids, :]  # (B, N_sub, D)
            pe = pe[token_ids]               # (N_sub, D)

        tokens = self.drop(tokens + pe.unsqueeze(0))

        intermediates: List[torch.Tensor] = []
        collect_layers = set(return_layers) if return_layers is not None else set()

        for i, block in enumerate(self.blocks):
            if self.grad_checkpointing and self.training:
                tokens = grad_ckpt.checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
            if i in collect_layers:
                intermediates.append(self.norm(tokens))

        tokens = self.norm(tokens)
        # proj_head is NOT applied here — forward() always returns embed_dim tokens.
        # Use project() to get pooled 2048-dim features for LP/downstream.

        if return_layers is not None:
            ordered = sorted(zip(collect_layers, intermediates), key=lambda t: t[0])
            return [t for _, t in ordered] + [tokens]

        return tokens
