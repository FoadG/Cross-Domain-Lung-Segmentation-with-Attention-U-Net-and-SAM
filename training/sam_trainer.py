"""
training/sam_trainer.py  [CORRECTED — adversarial review pass]

Fix C5/NaN: forward_with_box now returns raw logits (not sigmoid).
  - Removed logit reconstruction: torch.log(p/(1-p)) is deleted.
  - Loss functions receive real logits directly.
  - No float16 NaN possible.

Fix M2: gradient checkpointing passes deterministic=True and correct args.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data.dataset import SAMDataset, sam_collate_fn
from evaluation.metrics import compute_batch_dice, compute_batch_iou
from models.sam_wrapper import SAMFineTuner, load_sam, configure_sam_freeze_encoder
from models.lora_sam import inject_lora_into_sam, enable_gradient_checkpointing_sam, get_lora_state_dict
from prompt_generation.perturbation import perturb_box_batch
from training.losses import build_loss
from training.unet_trainer import EarlyStopping
from utils.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from utils.logging_utils import ExperimentLogger
from utils.memory_monitor import MemoryMonitor
from utils.reproducibility import get_worker_init_fn

logger = logging.getLogger(__name__)


def train_sam(
    df: pd.DataFrame,
    box_cache_df: pd.DataFrame,
    cfg: dict,
    checkpoint_dir: str,
    log_dir: str,
    device: Optional[torch.device] = None,
    seed: int = 42,
) -> str:
    """Fine-tune SAM (Freeze-Encoder or LoRA path based on cfg model name)."""
    from utils.reproducibility import seed_everything
    seed_everything(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    memory_monitor = MemoryMonitor(limit_gb=14.0)
    is_lora = "lora" in cfg["model"]["name"]
    experiment_name = "sam_lora" if is_lora else "sam_freeze"

    logger.info(f"SAM training: {'LoRA' if is_lora else 'Freeze-Encoder'}")

    sam_model, processor = load_sam(model_id=cfg["model"]["sam_model_id"], device=device)

    if is_lora:
        lora_cfg = cfg["model"]["lora"]
        sam_model = inject_lora_into_sam(
            sam_model,
            rank=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=lora_cfg.get("dropout", 0.05),
            target_module_name=lora_cfg["target_module_name"],
        )
        if cfg["training"].get("gradient_checkpointing", False):
            enable_gradient_checkpointing_sam(sam_model)
    else:
        configure_sam_freeze_encoder(sam_model)

    native_size = cfg.get("native_size", 512)
    use_cache = cfg["model"].get("cache_image_embeddings", False) and not is_lora
    cache_dir = cfg["model"].get("cache_dir") if use_cache else None

    sam_finetuner = SAMFineTuner(
        model=sam_model, processor=processor, device=device,
        use_cache=use_cache, cache_dir=cache_dir,
        multimask_output=cfg["training"].get("multimask_output", False),
    )

    if use_cache:
        train_paths = df[(df["source"] == "Shenzhen") & (df["split"] == "train")]["image_path"].tolist()
        with memory_monitor.track("precompute_embeddings"):
            sam_finetuner.precompute_embeddings(train_paths, native_size=native_size, batch_size=4)

    train_df = df[(df["source"] == "Shenzhen") & (df["split"] == "train")].reset_index(drop=True)
    val_df = df[(df["source"] == "Shenzhen") & (df["split"] == "val")].reset_index(drop=True)

    train_dataset = SAMDataset(train_df, box_cache_df, native_size, "train")
    val_dataset = SAMDataset(val_df, box_cache_df, native_size, "val")

    worker_init = get_worker_init_fn(seed)
    batch_size = cfg["training"]["batch_size"]
    grad_accum = cfg["training"].get("grad_accum_steps", 1)

    # PASS-2 fix #8: dedicated generator for the training sampler so the batch
    # ORDER is deterministic per (seed, epoch) and resume-stable. Without this,
    # shuffle drew from the global torch RNG, whose state is not checkpointed,
    # so a resumed run used a different batch order than an uninterrupted one —
    # making the perturbation fix insufficient for true reproducibility.
    train_gen = torch.Generator()
    train_gen.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=sam_collate_fn,
                              num_workers=cfg["training"]["num_workers"],
                              worker_init_fn=worker_init, pin_memory=False,
                              generator=train_gen)
    val_loader = DataLoader(val_dataset, batch_size=cfg["inference"]["batch_size"],
                            shuffle=False, collate_fn=sam_collate_fn,
                            num_workers=cfg["training"]["num_workers"],
                            worker_init_fn=worker_init, pin_memory=False)

    trainable_params = [p for p in sam_model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg["training"]["learning_rate"],
                      weight_decay=cfg["training"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["training"]["n_epochs"],
                                   eta_min=cfg["training"]["scheduler_eta_min"])

    criterion = build_loss(cfg["training"]["loss"], alpha=cfg["training"]["loss_alpha"])

    use_amp = cfg["training"].get("use_amp", True)
    scaler = GradScaler() if use_amp else None

    early_stop = EarlyStopping(
        patience=cfg["training"]["early_stopping_patience"],
        min_delta=cfg["training"]["early_stopping_min_delta"],
    )

    exp_logger = ExperimentLogger(log_dir, experiment_name)

    latest_ckpt = find_latest_checkpoint(checkpoint_dir, prefix="epoch_")
    start_epoch = 0
    if latest_ckpt:
        ckpt_data = load_checkpoint(latest_ckpt, sam_model, optimizer, scheduler,
                                     device=device, strict=False)
        start_epoch = ckpt_data.get("epoch", 0) + 1
        logger.info(f"Resuming from epoch {start_epoch}")

    best_val_dice = 0.0

    for epoch in range(start_epoch, cfg["training"]["n_epochs"]):
        # PASS-2 fix #8: make this epoch's batch order and torch ops (e.g.
        # decoder dropout) deterministic per (seed, epoch) and resume-stable.
        epoch_seed = seed + epoch
        torch.manual_seed(epoch_seed)
        train_gen.manual_seed(epoch_seed)
        sam_model.train()
        train_metrics = _run_sam_epoch(
            finetuner=sam_finetuner, loader=train_loader, criterion=criterion,
            optimizer=optimizer, scaler=scaler, grad_accum=grad_accum,
            native_size=native_size,
            perturb=cfg["augmentation"].get("box_perturbation", True),
            perturb_min=0, perturb_max=30, device=device,
            memory_monitor=memory_monitor,
            perturb_seed=seed, epoch=epoch,
        )

        val_metrics = _run_sam_eval(sam_finetuner, val_loader, native_size, device)
        scheduler.step()

        val_dice = val_metrics["dice"]
        is_best = val_dice > best_val_dice
        if is_best:
            best_val_dice = val_dice

        exp_logger.log_metrics(
            epoch=epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            lr=scheduler.get_last_lr()[0],
        )

        save_checkpoint(model=sam_model, optimizer=optimizer, epoch=epoch,
                        metrics=val_metrics, config=cfg, checkpoint_dir=checkpoint_dir,
                        filename=f"epoch_{epoch:04d}.pth", scheduler=scheduler,
                        scaler=scaler, is_best=is_best,
                        # Fix #5: persist only trainable weights (LoRA + decoder +
                        # prompt encoder); the frozen vision encoder is restored
                        # from the pretrained checkpoint on resume (strict=False).
                        state_dict_override=get_lora_state_dict(sam_model),
                        keep_last_n=cfg["training"].get("keep_last_n_checkpoints", 2))

        logger.info(
            f"Epoch {epoch:3d}/{cfg['training']['n_epochs']} | "
            f"train_loss={train_metrics['total_loss']:.4f} | "
            f"val_dice={val_dice:.4f} | best={best_val_dice:.4f}"
        )

        if early_stop.should_stop(val_dice):
            logger.info(f"Early stopping at epoch {epoch}.")
            break

    exp_logger.close()
    best_path = os.path.join(checkpoint_dir, "best_model.pth")
    return best_path


def _run_sam_epoch(
    finetuner: SAMFineTuner,
    loader: DataLoader,
    criterion,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    grad_accum: int,
    native_size: int,
    perturb: bool,
    perturb_min: int,
    perturb_max: int,
    device: torch.device,
    memory_monitor: MemoryMonitor,
    perturb_seed: Optional[int] = None,
    epoch: int = 0,
) -> Dict[str, float]:
    """
    One SAM training epoch.

    Fix C5/NaN: forward_with_box returns raw logits now.
    No logit reconstruction (torch.log(p/(1-p))) needed or used.
    Loss functions receive raw logits directly — no NaN possible.

    Fix #8: box perturbation uses a per-epoch seeded RNG
    (RandomState(perturb_seed + epoch)) so runs are reproducible AND a Colab
    resume reproduces the same perturbations for a given epoch. If perturb_seed
    is None, the seeded global np.random is used.
    """
    total_losses, dice_losses, bce_losses = [], [], []
    optimizer.zero_grad()

    # Per-epoch deterministic RNG for box perturbation (audit fix #8).
    perturb_rng = (
        np.random.RandomState(perturb_seed + epoch)
        if perturb_seed is not None else None
    )

    for step, batch in enumerate(loader):
        gt_masks = batch["gt_masks"].to(device)
        rgb_images = batch["rgb_images"]
        boxes = batch["boxes"]
        image_paths = batch["image_paths"]

        if perturb:
            boxes = perturb_box_batch(boxes, min_pixels=perturb_min,
                                       max_pixels=perturb_max, image_size=native_size,
                                       rng=perturb_rng)

        try:
            if scaler is not None:
                with autocast():
                    # forward_with_box now returns RAW LOGITS — no reconstruction needed
                    logits, _ = finetuner.forward_with_box(rgb_images, boxes, image_paths)
                    loss, components = criterion(logits, gt_masks)
                    loss = loss / grad_accum
                scaler.scale(loss).backward()
            else:
                logits, _ = finetuner.forward_with_box(rgb_images, boxes, image_paths)
                loss, components = criterion(logits, gt_masks)
                loss = loss / grad_accum
                loss.backward()

        except torch.cuda.OutOfMemoryError:
            logger.error(
                f"OOM at step {step}. VRAM={memory_monitor.get_current_vram_gb():.2f}GB. "
                "Skipping batch."
            )
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            continue

        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in finetuner.model.parameters() if p.requires_grad], 1.0
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in finetuner.model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
            optimizer.zero_grad()

        total_losses.append(components["total_loss"])
        dice_losses.append(components["dice_loss"])
        bce_losses.append(components["bce_loss"])

    return {
        "total_loss": float(np.mean(total_losses)) if total_losses else 0.0,
        "dice_loss": float(np.mean(dice_losses)) if dice_losses else 0.0,
        "bce_loss": float(np.mean(bce_losses)) if bce_losses else 0.0,
    }


@torch.no_grad()
def _run_sam_eval(
    finetuner: SAMFineTuner,
    loader: DataLoader,
    native_size: int,
    device: torch.device,
) -> Dict[str, float]:
    """Validate SAM. Applies sigmoid to logits for binary mask computation."""
    finetuner.model.eval()
    all_dice, all_iou = [], []

    for batch in loader:
        gt_masks = batch["gt_masks"].to(device)
        logits, _ = finetuner.forward_with_box(
            batch["rgb_images"], batch["boxes"], batch["image_paths"]
        )
        # Apply sigmoid here (logits → probabilities → binary)
        preds = (torch.sigmoid(logits) > 0.5).float()

        all_dice.extend(compute_batch_dice(preds, gt_masks))
        all_iou.extend(compute_batch_iou(preds, gt_masks))

    finetuner.model.train()
    return {
        "dice": float(np.mean(all_dice)) if all_dice else 0.0,
        "iou": float(np.mean(all_iou)) if all_iou else 0.0,
    }
