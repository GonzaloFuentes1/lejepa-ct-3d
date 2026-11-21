"""
LeJEPA Lightning Module — SSL pretraining with 4 configurable encoder modes.

Modes (selected via Hydra config `encoder.mode`):
    lejepa           : LeJEPA — JEPA prediction + SIGReg, no EMA.
    jepa_ema         : Classic I-JEPA — context encoder + EMA target encoder.
    lejepa_diff_local: Flow Matching over target patch embeddings (conditioned on context).
    lejepa_diff_global: Flow Matching over mean-pooled global volume embeddings.

Improvements over baseline:
    - Cross-attention predictor (mask tokens query context)
    - EMA cosine schedule (τ ramps from 0.996 → 0.9999, DINOv2 style)
    - LLRD (layer-wise LR decay, factor 0.75 per layer)
    - Mask curriculum (start with large blocks → small blocks)
    - Collapse monitoring (effective rank + mean pairwise cosine, every 500 steps)
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import pytorch_lightning as pl
from omegaconf import DictConfig

from models.encoder.vit3d import ViT3D
from models.encoder.predictor3d import Predictor3D
from models.encoder.score_net import FlowNetPatch, FlowNetGlobal
from models.ssl.sigreg import SIGReg
from models.ssl.ema import EMAEncoder
from utils.masking import sample_block_masks_3d


class LeJEPAModule(pl.LightningModule):
    """
    PyTorch Lightning module for LeJEPA SSL pretraining.

    Args:
        cfg: Hydra config with encoder, trainer, optimizer, and scheduler sub-configs.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.mode: str = cfg.encoder.mode

        enc_cfg = cfg.encoder
        in_channels = cfg.data.get("in_channels", 1)
        self.encoder = ViT3D(
            in_channels=in_channels,
            patch_size=enc_cfg.patch_size,
            embed_dim=enc_cfg.embed_dim,
            depth=enc_cfg.depth,
            num_heads=enc_cfg.num_heads,
            mlp_ratio=enc_cfg.mlp_ratio,
            drop_rate=enc_cfg.get("drop_rate", 0.0),
            attn_drop_rate=enc_cfg.get("attn_drop_rate", 0.0),
            volume_size=tuple(cfg.data.patch_size),
            grad_checkpointing=enc_cfg.get("grad_checkpointing", False),
            proj_dim=enc_cfg.get("proj_dim", None),
        )

        self.predictor: nn.Module | None = None
        self.target_encoder: EMAEncoder | None = None
        self.sigreg: SIGReg | None = None
        self.score_net: FlowNetPatch | FlowNetGlobal | None = None

        vol = tuple(cfg.data.patch_size)
        patch_size = enc_cfg.patch_size
        enc_dim  = enc_cfg.embed_dim                       # 1024 — raw token space
        pred_dim = enc_cfg.get("predictor_pred_dim", 384)  # narrow internal predictor dim

        if self.mode in ("lejepa", "jepa_ema"):
            self.predictor = Predictor3D(
                embed_dim=enc_dim,
                pred_dim=pred_dim,
                depth=enc_cfg.get("predictor_depth", 6),
                num_heads=enc_cfg.get("predictor_num_heads", 6),
                volume_size=vol,
                patch_size=patch_size,
                predictor_type=enc_cfg.get("predictor_type", "cross_attention"),
            )

        # SIGReg operates in encoder token space (enc_dim=1024)
        if self.mode != "jepa_ema":
            self.sigreg = SIGReg(
                embed_dim=enc_dim,
                num_proj=enc_cfg.get("sigreg_num_proj", 256),
            )

        if self.mode == "jepa_ema":
            self.target_encoder = EMAEncoder(
                self.encoder,
                momentum=enc_cfg.get("ema_momentum", 0.999),
            )

        if self.mode == "lejepa_diff_local":
            self.score_net = FlowNetPatch(
                embed_dim=enc_dim,
                depth=enc_cfg.get("score_depth", 4),
                num_heads=enc_cfg.num_heads,
            )

        if self.mode == "lejepa_diff_global":
            self.score_net = FlowNetGlobal(
                embed_dim=enc_dim,
                depth=enc_cfg.get("score_depth", 4),
                num_heads=enc_cfg.num_heads,
            )

    # ------------------------------------------------------------------
    # Mask curriculum: large blocks early → small blocks late
    # ------------------------------------------------------------------

    def _get_mask_scale_range(self, step: int, max_steps: int) -> Tuple[float, float]:
        """Cosine curriculum from easy (large blocks) to hard (small blocks)."""
        enc_cfg = self.cfg.encoder
        if not enc_cfg.get("mask_curriculum", False):
            rng = enc_cfg.get("block_scale_range", [0.1, 0.5])
            return (rng[0], rng[1])

        scale_start = list(enc_cfg.get("mask_scale_start", [0.20, 0.80]))
        scale_end   = list(enc_cfg.get("mask_scale_end",   [0.05, 0.30]))
        progress = min(step / max(1, max_steps), 1.0)
        s_min = scale_start[0] + (scale_end[0] - scale_start[0]) * progress
        s_max = scale_start[1] + (scale_end[1] - scale_start[1]) * progress
        return (s_min, s_max)

    # ------------------------------------------------------------------
    # Collapse monitoring (every 500 steps, no grad)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _log_collapse_metrics(self, emb: torch.Tensor) -> None:
        """Log effective rank and mean pairwise cosine (health/ section in W&B)."""
        if self.global_step % 200 != 0:
            return
        z = emb.float().detach()

        # Effective rank: subsample to ≤512 rows so SVD stays fast (O(n·d²))
        try:
            max_rows = 512
            z_sub = z[:max_rows] if z.size(0) > max_rows else z
            z_c = z_sub - z_sub.mean(0)
            _, S, _ = torch.linalg.svd(z_c, full_matrices=False)
            p = (S / S.sum().clamp(min=1e-9)).clamp(min=1e-9)
            eff_rank = torch.exp(-(p * p.log()).sum()).item()
        except Exception:
            eff_rank = 0.0

        # Mean pairwise cosine similarity (sample up to 256 pairs)
        z_norm = F.normalize(z, dim=-1)
        n = min(z_norm.size(0), 256)
        idx = torch.randperm(z_norm.size(0), device=z.device)[:n]
        z_s = z_norm[idx]
        cos_mat = z_s @ z_s.T
        off_diag = cos_mat[~torch.eye(n, dtype=torch.bool, device=z.device)]
        mean_cos = off_diag.mean().item()

        self.log("health/eff_rank",         eff_rank, on_step=True, prog_bar=False)
        self.log("health/mean_pairwise_cos", mean_cos, on_step=True, prog_bar=False)

    # ------------------------------------------------------------------
    # Training step dispatcher
    # ------------------------------------------------------------------

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        t_step_start = time.perf_counter()

        x = batch["image"]  # (B, C, D, H, W)

        enc_cfg = self.cfg.encoder
        patch_size = enc_cfg.patch_size
        vol = tuple(self.cfg.data.patch_size)
        grid_shape = (vol[0] // patch_size, vol[1] // patch_size, vol[2] // patch_size)
        max_steps  = self.cfg.trainer.get("max_steps", 100_000)

        block_scale_range = self._get_mask_scale_range(self.global_step, max_steps)

        ctx_ids, tgt_ids = sample_block_masks_3d(
            grid_shape=grid_shape,
            mask_ratio=enc_cfg.get("mask_ratio", 0.75),
            block_scale_range=block_scale_range,
            n_target_blocks=enc_cfg.get("n_target_blocks", 4),
            seed=self.global_step,
            device=x.device,
        )

        # Record CUDA events without synchronizing — elapsed_time() is called later in
        # on_train_batch_end so the GPU pipeline is never flushed mid-step.
        t_fwd_start = torch.cuda.Event(enable_timing=True)
        t_fwd_end   = torch.cuda.Event(enable_timing=True)
        t_fwd_start.record()
        t_step_wall = time.perf_counter()

        if self.mode == "lejepa":
            loss = self._step_lejepa(x, ctx_ids, tgt_ids)
        elif self.mode == "jepa_ema":
            loss = self._step_jepa_ema(x, ctx_ids, tgt_ids)
        elif self.mode == "lejepa_diff_local":
            loss = self._step_lejepa_diff_local(x, ctx_ids, tgt_ids)
        elif self.mode == "lejepa_diff_global":
            loss = self._step_lejepa_diff_global(x)
        else:
            raise ValueError(f"Unknown encoder mode: {self.mode}")

        t_fwd_end.record()

        self.log("train/total_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        opt = self.optimizers()
        if opt is not None:
            self.log("train/lr", opt.param_groups[0]["lr"], on_step=True)

        # Stash events for logging after the optimizer step (avoids mid-step GPU sync).
        self._perf_events = (t_fwd_start, t_fwd_end, t_step_start, t_step_wall)

        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> None:
        x = batch["image"]
        enc_cfg = self.cfg.encoder
        patch_size = enc_cfg.patch_size
        vol = tuple(self.cfg.data.patch_size)
        grid_shape = (vol[0] // patch_size, vol[1] // patch_size, vol[2] // patch_size)
        ctx_ids, tgt_ids = sample_block_masks_3d(
            grid_shape=grid_shape,
            mask_ratio=enc_cfg.get("mask_ratio", 0.75),
            seed=batch_idx,
            device=x.device,
        )
        with torch.no_grad():
            ctx = self.encoder(x, ctx_ids)
            tgt = self.encoder(x, tgt_ids)
            if self.predictor is not None:
                pred = self.predictor(ctx, ctx_ids, tgt_ids)
                val_loss = F.mse_loss(pred, tgt)
            else:
                val_loss = torch.tensor(0.0, device=x.device)
        self.log("val/pred_loss", val_loss, on_epoch=True, sync_dist=True)

    # ------------------------------------------------------------------
    # SSL mode implementations
    # ------------------------------------------------------------------

    def _step_lejepa(
        self,
        x: torch.Tensor,
        ctx_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
    ) -> torch.Tensor:
        """LeJEPA: JEPA prediction loss + SIGReg (no EMA, no stop-gradient, no VCReg)."""
        # Dos llamadas separadas con shapes fijas → CUDAGraphs activos con torch.compile
        ctx_emb = self.encoder(x, ctx_ids)                       # (B, N_ctx, embed_dim)
        tgt_emb = self.encoder(x, tgt_ids)                       # (B, N_tgt, embed_dim)
        all_emb = torch.cat([ctx_emb, tgt_emb], dim=1)          # para SIGReg y logging

        pred_emb = self.predictor(ctx_emb, ctx_ids, tgt_ids)  # (B, N_tgt, out_dim)

        pred_loss   = F.mse_loss(pred_emb, tgt_emb.detach())
        flat_emb    = all_emb.flatten(0, 1)  # (B*N, out_dim)
        sigreg_loss = self.sigreg(flat_emb)

        lam  = self.cfg.encoder.get("sigreg_lambda", 0.1)
        loss = pred_loss + lam * sigreg_loss

        with torch.no_grad():
            cos_sim  = F.cosine_similarity(pred_emb.flatten(0, 1), tgt_emb.detach().flatten(0, 1)).mean()
            emb_std  = flat_emb.std(dim=0).mean()
            emb_norm = flat_emb.norm(dim=-1).mean()

        n_total = flat_emb.size(0) * (dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1)
        self.log("train/pred_loss",   pred_loss,              on_step=True, prog_bar=True)
        self.log("train/sigreg_loss", sigreg_loss / n_total,  on_step=True)
        self.log("train/cosine_sim",  cos_sim,                on_step=True, prog_bar=True)
        self.log("health/emb_std",    emb_std,     on_step=True)
        self.log("health/emb_norm",   emb_norm,    on_step=True)
        self._log_collapse_metrics(flat_emb)
        return loss

    def _step_jepa_ema(
        self,
        x: torch.Tensor,
        ctx_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Classic I-JEPA: context encoder (gradient) + EMA target encoder (no-grad)."""
        ctx_emb  = self.encoder(x, ctx_ids)                      # (B, N_ctx, D)
        pred_emb = self.predictor(ctx_emb, ctx_ids, tgt_ids)     # (B, N_tgt, D)

        with torch.no_grad():
            tgt_emb = self.target_encoder(x, tgt_ids)            # (B, N_tgt, D)

        loss = F.mse_loss(pred_emb, tgt_emb)

        with torch.no_grad():
            cos_sim = F.cosine_similarity(pred_emb.flatten(0, 1), tgt_emb.flatten(0, 1)).mean()
            tgt_std = tgt_emb.flatten(0, 1).std(dim=0).mean()
            ctx_std = ctx_emb.flatten(0, 1).std(dim=0).mean()

        self.log("train/pred_loss",    loss,    on_step=True, prog_bar=True)
        self.log("train/cosine_sim",   cos_sim, on_step=True, prog_bar=True)
        self.log("health/tgt_emb_std", tgt_std, on_step=True)
        self.log("health/ctx_emb_std", ctx_std, on_step=True)
        self._log_collapse_metrics(ctx_emb.flatten(0, 1))
        return loss

    def _step_lejepa_diff_local(
        self,
        x: torch.Tensor,
        ctx_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
    ) -> torch.Tensor:
        """LeJEPA + local Flow Matching: velocity field noise→target conditioned on context."""
        B = x.size(0)
        all_ids = torch.cat([ctx_ids, tgt_ids])
        all_emb = self.encoder(x, all_ids)
        n_ctx   = ctx_ids.shape[0]
        ctx_emb = all_emb[:, :n_ctx]
        tgt_emb = all_emb[:, n_ctx:]

        t  = self.score_net.schedule.sample_t(B, x.device)
        x0 = torch.randn_like(tgt_emb)
        x_t, velocity = self.score_net.schedule.interpolate(x0, tgt_emb.detach(), t)
        v_pred = self.score_net(x_t, t, ctx_emb)
        flow_loss = F.mse_loss(v_pred, velocity)

        flat_emb    = all_emb.flatten(0, 1)
        sigreg_loss = self.sigreg(flat_emb)

        lam  = self.cfg.encoder.get("sigreg_lambda", 0.1)
        loss = flow_loss + lam * sigreg_loss

        with torch.no_grad():
            v_cos   = F.cosine_similarity(v_pred.flatten(0, 1), velocity.flatten(0, 1)).mean()
            ctx_std = ctx_emb.flatten(0, 1).std(dim=0).mean()

        n_total = flat_emb.size(0) * (dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1)
        self.log("train/flow_loss",        flow_loss,              on_step=True, prog_bar=True)
        self.log("train/sigreg_loss",      sigreg_loss / n_total,  on_step=True)
        self.log("train/velocity_cos_sim", v_cos,                  on_step=True, prog_bar=True)
        self.log("health/ctx_emb_std",     ctx_std,                on_step=True)
        self._log_collapse_metrics(flat_emb)
        return loss

    def _step_lejepa_diff_global(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """LeJEPA + global Flow Matching: velocity field on mean-pooled embedding."""
        B = x.size(0)
        all_emb = self.encoder(x)
        z1 = all_emb.mean(dim=1)

        t  = self.score_net.schedule.sample_t(B, x.device)
        x0 = torch.randn_like(z1)
        x_t, velocity = self.score_net.schedule.interpolate(x0, z1.detach(), t)
        v_pred = self.score_net(x_t, t)
        flow_loss = F.mse_loss(v_pred, velocity)

        flat_emb    = all_emb.flatten(0, 1)
        sigreg_loss = self.sigreg(flat_emb)

        lam  = self.cfg.encoder.get("sigreg_lambda", 0.1)
        loss = flow_loss + lam * sigreg_loss

        with torch.no_grad():
            v_cos    = F.cosine_similarity(v_pred, velocity).mean()
            emb_std  = flat_emb.std(dim=0).mean()
            emb_norm = z1.norm(dim=-1).mean()

        n_total = flat_emb.size(0) * (dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1)
        self.log("train/flow_loss",        flow_loss,              on_step=True, prog_bar=True)
        self.log("train/sigreg_loss",      sigreg_loss / n_total,  on_step=True)
        self.log("train/velocity_cos_sim", v_cos,                  on_step=True, prog_bar=True)
        self.log("health/emb_std",         emb_std,                on_step=True)
        self.log("health/emb_norm",        emb_norm,               on_step=True)
        self._log_collapse_metrics(flat_emb)
        return loss

    # ------------------------------------------------------------------
    # EMA update (must be in on_train_batch_end, not training_step)
    # ------------------------------------------------------------------

    def on_train_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        # Log perf metrics here (after optimizer step) so synchronize() never blocks the
        # training step itself. Only flush every 50 steps to keep overhead low.
        if self.global_step % 50 == 0 and hasattr(self, "_perf_events"):
            t_fwd_start, t_fwd_end, t_step_start, t_step_wall = self._perf_events
            torch.cuda.synchronize()
            fwd_ms  = t_fwd_start.elapsed_time(t_fwd_end)
            step_ms = (time.perf_counter() - t_step_start) * 1000
            wall_ms = (time.perf_counter() - t_step_wall) * 1000
            data_ms = max(step_ms - wall_ms, 0)
            bwd_ms  = max(wall_ms - fwd_ms, 0)
            self.log("perf/t_forward_ms",  fwd_ms,  on_step=True, prog_bar=True)
            self.log("perf/t_data_ms",     data_ms, on_step=True, prog_bar=True)
            self.log("perf/t_backward_ms", bwd_ms,  on_step=True)
            self.log("perf/t_step_ms",     step_ms, on_step=True)
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            self.log(f"perf/rank{rank}/t_forward_ms", fwd_ms, on_step=True)

        if self.mode == "jepa_ema" and self.target_encoder is not None:
            enc_cfg   = self.cfg.encoder
            max_steps = self.cfg.trainer.get("max_steps", 100_000)
            if enc_cfg.get("ema_momentum_start") is not None:
                # Cosine schedule: τ ramps from base → final
                m = self.target_encoder.update_cosine(
                    source=self.encoder,
                    step=self.global_step,
                    max_steps=max_steps,
                    base_m=enc_cfg.get("ema_momentum_start", 0.996),
                    final_m=enc_cfg.get("ema_momentum_end", 0.9999),
                )
                self.log("train/ema_momentum", m, on_step=True)
            else:
                # Legacy fixed momentum
                self.target_encoder.update(self.encoder)

    # ------------------------------------------------------------------
    # Optimizer with LLRD
    # ------------------------------------------------------------------

    def _build_llrd_param_groups(
        self,
        base_lr: float,
        weight_decay: float,
        layer_decay: float,
    ) -> List[Dict]:
        """
        Layer-wise LR decay for the ViT encoder.

        Layer 0 (patch_embed) gets LR × decay^(depth+1).
        Layer i (blocks[i]) gets LR × decay^(depth-i).
        Encoder norm, predictor, sigreg, vcreg: full LR.
        LayerNorm / bias params: no weight decay.
        """
        depth = len(self.encoder.blocks)
        groups: List[Dict] = []

        # Patch embedding — deepest, lowest LR
        groups.append({
            "params": list(self.encoder.patch_embed.parameters()),
            "lr": base_lr * (layer_decay ** (depth + 1)),
            "weight_decay": weight_decay,
            "name": "patch_embed",
        })

        # Transformer blocks — layer-wise decay
        for i, block in enumerate(self.encoder.blocks):
            lr_scale = layer_decay ** (depth - i)
            groups.append({
                "params": list(block.parameters()),
                "lr": base_lr * lr_scale,
                "weight_decay": weight_decay,
                "name": f"encoder_block_{i}",
            })

        # Encoder norm + proj_head — full LR, no weight decay
        norm_params = list(self.encoder.norm.parameters())
        if self.encoder.proj_head is not None:
            norm_params += list(self.encoder.proj_head.parameters())
        groups.append({
            "params": norm_params,
            "lr": base_lr,
            "weight_decay": 0.0,
            "name": "encoder_norm_proj",
        })

        # Everything else (predictor, score_net, sigreg) — full LR
        other_params: List[nn.Parameter] = []
        for m in [m for m in [self.predictor, self.score_net, self.sigreg] if m is not None]:
            other_params.extend(m.parameters())
        if other_params:
            groups.append({
                "params": other_params,
                "lr": base_lr,
                "weight_decay": weight_decay,
                "name": "rest",
            })

        return groups

    def configure_optimizers(self) -> Dict:
        opt_cfg      = self.cfg.get("optimizer", {})
        lr           = opt_cfg.get("lr", 1.5e-4)
        weight_decay = opt_cfg.get("weight_decay", 0.05)
        betas        = tuple(opt_cfg.get("betas", [0.9, 0.95]))
        layer_decay  = opt_cfg.get("layer_decay", None)

        if layer_decay is not None and layer_decay < 1.0:
            param_groups = self._build_llrd_param_groups(lr, weight_decay, layer_decay)
        else:
            param_groups = self.parameters()

        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay, betas=betas)

        sched_cfg   = self.cfg.get("scheduler", {})
        warmup_steps = sched_cfg.get("warmup_steps", 2000)
        max_steps    = self.cfg.trainer.get("max_steps", 100_000)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
