"""Tests for ViT3D encoder, Predictor3D, and masking."""

import torch
import pytest
from models.encoder.vit3d import ViT3D, PatchEmbed3D
from models.encoder.predictor3d import Predictor3D
from utils.masking import sample_block_masks_3d, make_3d_sinusoidal_pos_embed


def _tiny_vit():
    return ViT3D(in_channels=1, patch_size=16, embed_dim=96, depth=2, num_heads=3,
                 volume_size=(64, 64, 64))


def test_patch_embed_shape():
    pe = PatchEmbed3D(in_channels=1, patch_size=16, embed_dim=96)
    x = torch.randn(2, 1, 64, 64, 64)
    tokens, grid = pe(x)
    assert grid == (4, 4, 4)
    assert tokens.shape == (2, 64, 96)


def test_vit3d_full_forward():
    enc = _tiny_vit()
    x = torch.randn(2, 1, 64, 64, 64)
    out = enc(x)
    assert out.shape == (2, 64, 96)


def test_vit3d_partial_forward():
    enc = _tiny_vit()
    x = torch.randn(2, 1, 64, 64, 64)
    ctx_ids, tgt_ids = sample_block_masks_3d((4, 4, 4), seed=0)
    ctx_emb = enc(x, ctx_ids)
    tgt_emb = enc(x, tgt_ids)
    assert ctx_emb.shape[0] == 2
    assert ctx_emb.shape[-1] == 96
    assert ctx_emb.shape[1] + tgt_emb.shape[1] == 64


def test_predictor3d_shape():
    enc = _tiny_vit()
    pred = Predictor3D(embed_dim=96, depth=2, num_heads=3,
                       volume_size=(64, 64, 64), patch_size=16)
    x = torch.randn(2, 1, 64, 64, 64)
    ctx_ids, tgt_ids = sample_block_masks_3d((4, 4, 4), seed=1)
    ctx_emb = enc(x, ctx_ids)
    tgt_emb = enc(x, tgt_ids)
    pred_emb = pred(ctx_emb, ctx_ids, tgt_ids)
    assert pred_emb.shape == tgt_emb.shape, f"{pred_emb.shape} != {tgt_emb.shape}"


def test_3d_sinusoidal_pe():
    pe = make_3d_sinusoidal_pos_embed(embed_dim=96, grid_shape=(4, 4, 4))
    assert pe.shape == (64, 96)
    assert not pe.isnan().any()


def test_mask_coverage():
    grid = (8, 8, 8)
    ctx, tgt = sample_block_masks_3d(grid, mask_ratio=0.75, seed=42)
    total = 8 * 8 * 8
    assert ctx.numel() + tgt.numel() == total
    assert len(set(ctx.tolist()) & set(tgt.tolist())) == 0, "Overlap between ctx and tgt"


def test_vit3d_param_count():
    enc = ViT3D(embed_dim=384, depth=12, num_heads=6, volume_size=(128, 128, 128))
    n = sum(p.numel() for p in enc.parameters())
    assert n < 50_000_000, f"Encoder has {n/1e6:.1f}M params, expected < 50M"
    print(f"ViT3D params: {n/1e6:.2f}M")
