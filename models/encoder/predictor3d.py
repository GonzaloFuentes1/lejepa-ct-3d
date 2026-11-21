"""
Narrow ViT predictor for JEPA patch prediction.

Takes context token embeddings and predicts the embeddings of masked
(target) tokens.

Two architectures (selected via predictor_type):
    cross_attention  : Mask tokens query context via cross-attention (default,
                       matches the LeJEPA paper). More efficient and cleaner
                       ctx→tgt information flow.
    self_attention   : Legacy concat + self-attention (I-JEPA style).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import make_3d_sinusoidal_pos_embed


class Predictor3D(nn.Module):
    """
    Narrow Transformer predictor for JEPA target prediction.

    Operates in a narrow internal dim (pred_dim, default 384) regardless of
    the encoder embed_dim (1024). Input/output projections bridge the two spaces.
    This keeps the predictor small (~16M params) while the encoder is large (307M).

    Args:
        embed_dim:  Encoder token dimension (e.g. 1024 for ViT-L).
        pred_dim:   Internal predictor dimension (narrow, e.g. 384).
        depth:      Number of transformer layers (6).
        num_heads:  Attention heads inside the predictor.
        volume_size, patch_size: For positional embedding precomputation.
        predictor_type: "cross_attention" (default) or "self_attention".
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        pred_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        volume_size: Tuple[int, int, int] = (96, 160, 160),
        patch_size: int = 16,
        predictor_type: str = "cross_attention",
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.pred_dim  = pred_dim
        self.predictor_type = predictor_type

        # Project encoder tokens into narrow predictor space
        self.input_proj  = nn.Linear(embed_dim, pred_dim)
        # Project predictor output back to encoder token space
        self.output_proj = nn.Linear(pred_dim, embed_dim)

        # Learnable mask token in predictor space
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        mlp_dim = int(pred_dim * mlp_ratio)
        if predictor_type == "cross_attention":
            self.blocks = nn.ModuleList([
                _CrossPredictorBlock(pred_dim, num_heads, mlp_dim)
                for _ in range(depth)
            ])
        else:
            self.blocks = nn.ModuleList([
                _PredictorBlock(pred_dim, num_heads, mlp_dim)
                for _ in range(depth)
            ])
        self.norm = nn.LayerNorm(pred_dim)

        # Precompute PE in pred_dim space
        Gd = volume_size[0] // patch_size
        Gh = volume_size[1] // patch_size
        Gw = volume_size[2] // patch_size
        if pred_dim % 6 == 0:
            pe = make_3d_sinusoidal_pos_embed(pred_dim, (Gd, Gh, Gw))
        else:
            pe = torch.zeros(Gd * Gh * Gw, pred_dim)
        self.register_buffer("pos_embed", pe)  # (N_total, pred_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        ctx_tokens: torch.Tensor,
        ctx_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        prompt_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict target token embeddings from context tokens.

        Args:
            ctx_tokens:  (B, N_ctx, D) context encoder outputs.
            ctx_ids:     (N_ctx,) flat token indices of context positions.
            tgt_ids:     (N_tgt,) flat token indices of target positions.
            prompt_emb:  (B, D) optional prompt — added to every target mask token.
        Returns:
            pred_tokens: (B, N_tgt, D) predicted target embeddings.
        """
        B = ctx_tokens.size(0)
        pe = self.pos_embed.to(ctx_tokens.device)

        # Project encoder tokens (embed_dim) → predictor space (pred_dim)
        ctx_proj = self.input_proj(ctx_tokens)             # (B, N_ctx, pred_dim)
        ctx_with_pe = ctx_proj + pe[ctx_ids].unsqueeze(0)  # (B, N_ctx, pred_dim)

        # Mask tokens for target positions (already in pred_dim)
        N_tgt = tgt_ids.size(0)
        tgt_tokens = self.mask_token.expand(B, N_tgt, -1).clone()  # (B, N_tgt, pred_dim)
        tgt_with_pe = tgt_tokens + pe[tgt_ids].unsqueeze(0)         # (B, N_tgt, pred_dim)

        if prompt_emb is not None:
            tgt_with_pe = tgt_with_pe + self.input_proj(prompt_emb).unsqueeze(1)

        if self.predictor_type == "cross_attention":
            tgt = tgt_with_pe
            for block in self.blocks:
                tgt = block(tgt, ctx_with_pe)
            tgt = self.norm(tgt)                           # (B, N_tgt, pred_dim)
        else:
            x = torch.cat([ctx_with_pe, tgt_with_pe], dim=1)
            for block in self.blocks:
                x = block(x)
            x = self.norm(x)
            tgt = x[:, ctx_ids.size(0):, :]               # (B, N_tgt, pred_dim)

        # Project back to encoder token space (embed_dim)
        return self.output_proj(tgt)                       # (B, N_tgt, embed_dim)


class _CrossPredictorBlock(nn.Module):
    """
    Cross-attention predictor block (LeJEPA paper style).

    Target tokens:
      1. Self-attention among themselves (aggregate target context)
      2. Cross-attention to context tokens (fetch relevant information)
      3. MLP

    This enforces clean ctx→tgt information flow and is more efficient than
    self-attention over the full concatenated sequence.
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int) -> None:
        super().__init__()
        # Self-attention on target tokens
        self.norm_tgt1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # Cross-attention: tgt queries, ctx keys/values
        self.norm_tgt2 = nn.LayerNorm(embed_dim)
        self.norm_ctx  = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # MLP
        self.norm_tgt3 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )

    def forward(self, tgt: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        # 1. Self-attention among target tokens
        t = self.norm_tgt1(tgt)
        tgt = tgt + self.self_attn(t, t, t)[0]
        # 2. Cross-attention: target queries attend to context
        tgt = tgt + self.cross_attn(
            self.norm_tgt2(tgt),
            self.norm_ctx(ctx),
            self.norm_ctx(ctx),
        )[0]
        # 3. MLP
        tgt = tgt + self.mlp(self.norm_tgt3(tgt))
        return tgt


class _PredictorBlock(nn.Module):
    """Legacy self-attention predictor block (I-JEPA style, concat ctx+tgt)."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x
