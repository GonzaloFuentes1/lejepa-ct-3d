"""EMA (Exponential Moving Average) target encoder for jepa_ema mode."""

from __future__ import annotations

import copy
import torch
import torch.nn as nn


class EMAEncoder(nn.Module):
    """
    EMA copy of an encoder. No gradients flow through this module.

    Used in jepa_ema mode (classic I-JEPA) as the target encoder.
    Must be updated after each optimizer step via .update(), NOT inside
    training_step — use on_train_batch_end in the LightningModule.

    Args:
        source: The encoder module to copy.
        momentum: EMA decay coefficient τ. Typical: 0.996–0.999.
            Higher → slower update, more stable targets.
    """

    def __init__(self, source: nn.Module, momentum: float = 0.999) -> None:
        super().__init__()
        self.momentum = momentum
        self.encoder = copy.deepcopy(source)
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, source: nn.Module) -> None:
        """
        Update EMA weights. Call in on_train_batch_end, not training_step.

        Args:
            source: The online encoder whose weights are the source of truth.
        """
        m = self.momentum
        for ema_p, src_p in zip(
            self.encoder.parameters(), source.parameters()
        ):
            ema_p.data.mul_(m).add_(src_p.data, alpha=1.0 - m)

    @torch.no_grad()
    def update_cosine(
        self,
        source: nn.Module,
        step: int,
        max_steps: int,
        base_m: float = 0.996,
        final_m: float = 0.9999,
    ) -> float:
        """
        EMA update with cosine momentum schedule (DINOv2 / MAE-v2 style).

        τ(t) = τ_end - (τ_end - τ_start) × (cos(π·t/T) + 1) / 2

        Starts at base_m (fast updates when representations are chaotic)
        and ramps up to final_m (stable targets as representations converge).

        Returns:
            Current momentum value (for logging).
        """
        import math
        progress = min(step / max(1, max_steps), 1.0)
        m = final_m - (final_m - base_m) * (math.cos(math.pi * progress) + 1) / 2
        for ema_p, src_p in zip(self.encoder.parameters(), source.parameters()):
            ema_p.data.mul_(m).add_(src_p.data, alpha=1.0 - m)
        return m

    def forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)
