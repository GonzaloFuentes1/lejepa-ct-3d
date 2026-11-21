"""
SSL pretraining entry point — CVPR 2026 CT Foundation Model.

Uso:
    # LeJEPA + SIGReg en FLARE-CT coreset (default)
    python scripts/pretrain.py

    # Classic I-JEPA con EMA
    python scripts/pretrain.py encoder=jepa_ema

    # LeJEPA + Flow Matching local
    python scripts/pretrain.py encoder=lejepa_diff_local

    # LeJEPA + Flow Matching global
    python scripts/pretrain.py encoder=lejepa_diff_global

    # Override número de GPUs
    python scripts/pretrain.py trainer.devices=2

    # Reanudar desde checkpoint
    python scripts/pretrain.py resume=checkpoints/last.ckpt
"""

from __future__ import annotations

import os
import warnings

# Suprimir warnings de librerías que no podemos cambiar
warnings.filterwarnings("ignore", message=".*cuda.cudart.*deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*LeafSpec.*")
warnings.filterwarnings("ignore", message=".*treespec.*")
warnings.filterwarnings("ignore", message=".*lru_cache.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*anonymous setting.*", category=UserWarning)
os.environ.setdefault("WANDB_ANONYMOUS", "never")
warnings.filterwarnings("ignore", message=".*srun.*command is available.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*litlogger.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Online softmax is disabled.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*AccumulateGrad node.*stream.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Orientationd.*labels.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*warn_deprecated.*", category=FutureWarning)

import torch

torch.set_float32_matmul_precision("high")   # TF32 matmul en H100/A100
torch.backends.cuda.matmul.allow_tf32 = True  # ~1.5× speedup en matmul
torch.backends.cudnn.allow_tf32 = True        # ~1.2× speedup en conv
torch.backends.cudnn.benchmark = True         # autotuner de kernels para shapes fijas

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import logging
logging.getLogger("lightning.pytorch.trainer.connectors.signal_connector").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from data.ct_dataset import build_ct_dataset
from models.encoder.lejepa_module import LeJEPAModule
from utils.pca_viz_callback import PCAVizCallback
from utils.lp_eval_callback import LPEvalCallback
from utils.seg_pca_viz_callback import SegPCAVizCallback


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)

    train_ds = build_ct_dataset(cfg.data, split="train")
    val_ds   = build_ct_dataset(cfg.data, split="val")

    num_workers    = cfg.data.num_workers
    prefetch       = cfg.data.get("prefetch_factor", 2)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size_per_gpu,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch if num_workers > 0 else None,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size_per_gpu,
        shuffle=False,
        num_workers=min(num_workers, 4),          # val no necesita tantos workers
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    module = LeJEPAModule(cfg)

    # torch.compile con reduce-overhead: activa CUDAGraphs → GPU al 100%
    # Requiere shapes fijas en el encoder — garantizado por las dos llamadas separadas
    # en _step_lejepa (ctx_ids y tgt_ids tienen longitud constante por budgets fijos)
    if cfg.get("compile", True):
        module.encoder = torch.compile(module.encoder, mode="reduce-overhead")
        if module.predictor is not None:
            module.predictor = torch.compile(module.predictor, mode="reduce-overhead")

    pca_every = cfg.get("pca_viz_every_n_steps", 1000)
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.checkpoint.dirpath,
            every_n_train_steps=cfg.checkpoint.every_n_train_steps,
            save_top_k=cfg.checkpoint.save_top_k,
            monitor="train/total_loss",
            mode="min",
            filename=cfg.checkpoint.filename,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=10),   # actualiza cada 10 batches
        PCAVizCallback(
            val_loader=val_loader,
            every_n_steps=pca_every,
            patch_size=cfg.encoder.patch_size,
        ),
        # LPEvalCallback desactivado temporalmente (GPU desync)
        # LPEvalCallback(
        #     data_dir=cfg.get(
        #         "lp_data_dir",
        #         "$LEJEPA_DATA_ROOT/cvpr26_downstream/AMOS-clf-tr-val",
        #     ),
        #     every_n_epochs=cfg.get("lp_eval_every_n_epochs", 3),
        #     max_samples=cfg.get("lp_max_samples", None),
        # ),
        SegPCAVizCallback(
            every_n_steps=pca_every,
            patch_size=cfg.encoder.patch_size,
        ),
    ]

    logger = (
        WandbLogger(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            entity=cfg.wandb.entity or None,
        )
        if cfg.wandb.project
        else True
    )

    trainer = pl.Trainer(
        max_steps=cfg.trainer.max_steps,
        precision=cfg.trainer.precision,
        devices=cfg.trainer.devices,
        num_nodes=cfg.trainer.num_nodes,
        strategy=cfg.trainer.strategy,
        sync_batchnorm=cfg.trainer.sync_batchnorm,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        val_check_interval=cfg.trainer.val_check_interval,
        limit_train_batches=cfg.trainer.get("limit_train_batches", 1.0),
        limit_val_batches=cfg.trainer.get("limit_val_batches", 1.0),
        check_val_every_n_epoch=None,   # usar steps globales, no epochs
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
    )

    trainer.fit(module, train_loader, val_loader, ckpt_path=cfg.resume)


if __name__ == "__main__":
    main()
