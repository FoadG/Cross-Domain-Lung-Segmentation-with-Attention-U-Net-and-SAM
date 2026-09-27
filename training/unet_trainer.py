"""
training/unet_trainer.py

U-Net training logic with:
  - 5-fold cross-validation for robustness assessment (RQ question components)
  - Final model training on full train+val (used for prompt generation)
  - Early stopping on per-image mean Dice
  - Checkpoint saving every epoch
  - Separate metric logging for Dice and BCE components

Audit fix: CRITICAL-12 — specifies that prompt-generating U-Net uses full train+val.
Audit fix: CRITICAL-18 — early stopping monitors per-image Dice, not pixel accuracy.
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
from torch.utils.data import DataLoader, Subset

from data.dataset import UNetDataset
from data.split import get_fold_indices
from data.transforms import get_unet_train_transforms, get_unet_val_transforms
from evaluation.metrics import compute_batch_dice, compute_batch_iou
from models.unet import build_unet
from training.losses import build_loss
from utils.checkpoint import (
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from utils.logging_utils import ExperimentLogger
from utils.reproducibility import get_worker_init_fn

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping monitor based on validation Dice.

    Args:
        patience: Number of epochs without improvement before stopping.
        min_delta: Minimum improvement to reset patience counter.
    """

    def __init__(self, patience: int = 15, min_delta: float = 0.001) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._best_score = -float("inf")
        self._counter = 0

    def should_stop(self, score: float) -> bool:
        """
        Args:
            score: Current epoch validation Dice (higher is better).

        Returns:
            True if training should stop.
        """
        if score > self._best_score + self.min_delta:
            self._best_score = score
            self._counter = 0
            return False
        else:
            self._counter += 1
            return self._counter >= self.patience

    @property
    def best_score(self) -> float:
        return self._best_score


def train_unet_kfold(
    df: pd.DataFrame,
    cfg: dict,
    norm_stats: dict,
    checkpoint_base_dir: str,
    log_dir: str,
    device: Optional[torch.device] = None,
    seed: int = 42,
) -> Dict[str, List[float]]:
    """
    Run 5-fold cross-validation for U-Net robustness assessment.

    Each fold trains a fresh U-Net and evaluates on the held-out fold.
    Results are summarized as mean ± std across folds.

    Note: The checkpoints from this function are for EVALUATION only.
    The prompt-generating U-Net is trained separately by train_unet_final().

    Args:
        df: Full DataFrame with split column populated.
        cfg: U-Net configuration dict.
        norm_stats: Pre-computed normalization statistics.
        checkpoint_base_dir: Directory to save per-fold checkpoints.
        log_dir: Directory for metric logs.
        device: Computation device.
        seed: Random seed.

    Returns:
        Dict with 'dice', 'iou' lists (one value per fold).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get K-fold indices into Shenzhen train+val subset
    shenzhen_trainval = df[
        (df["source"] == "Shenzhen") & (df["split"].isin(["train", "val"]))
    ].reset_index(drop=True)

    fold_indices = get_fold_indices(
        df, n_folds=cfg.get("n_folds", 5), seed=seed
    )

    fold_results = {"dice": [], "iou": []}

    for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
        logger.info(f"=== Training Fold {fold_idx + 1}/{len(fold_indices)} ===")

        fold_train = shenzhen_trainval.iloc[train_idx].reset_index(drop=True)
        fold_val = shenzhen_trainval.iloc[val_idx].reset_index(drop=True)

        fold_checkpoint_dir = os.path.join(
            checkpoint_base_dir, f"fold_{fold_idx}"
        )

        best_dice, best_iou = _train_single_unet(
            train_df=fold_train,
            val_df=fold_val,
            cfg=cfg,
            norm_stats=norm_stats,
            checkpoint_dir=fold_checkpoint_dir,
            log_dir=log_dir,
            experiment_name=f"unet_fold{fold_idx}",
            device=device,
            seed=seed,
        )

        fold_results["dice"].append(best_dice)
        fold_results["iou"].append(best_iou)

        logger.info(
            f"Fold {fold_idx + 1} complete: "
            f"Dice={best_dice:.4f}, IoU={best_iou:.4f}"
        )

    logger.info("=" * 50)
    logger.info(
        f"K-Fold CV Results ({len(fold_indices)} folds):\n"
        f"  Dice: {np.mean(fold_results['dice']):.4f} ± "
        f"{np.std(fold_results['dice']):.4f}\n"
        f"  IoU:  {np.mean(fold_results['iou']):.4f} ± "
        f"{np.std(fold_results['iou']):.4f}"
    )
    logger.info("=" * 50)

    return fold_results


def train_unet_final(
    df: pd.DataFrame,
    cfg: dict,
    norm_stats: dict,
    checkpoint_dir: str,
    log_dir: str,
    device: Optional[torch.device] = None,
    seed: int = 42,
) -> str:
    """
    Train the final U-Net model on the full Shenzhen train+val subset.

    This model's checkpoint is the one used by prompt_generation/ to
    produce bounding boxes for SAM training. It uses a larger training set
    than any single fold.

    Audit fix: CRITICAL-12 — train+val (85% of Shenzhen) for final model.

    Args:
        df: Full DataFrame with split column populated.
        cfg: U-Net configuration dict.
        norm_stats: Pre-computed normalization statistics.
        checkpoint_dir: Directory to save checkpoints.
        log_dir: Directory for metric logs.
        device: Computation device.
        seed: Random seed.

    Returns:
        Path to the best final model checkpoint.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use train+val (NOT test) for the final model — more data than any fold.
    train_df = df[
        (df["source"] == "Shenzhen") & (df["split"].isin(["train", "val"]))
    ].reset_index(drop=True)

    # NOTE (audit fix #6): the val split is a SUBSET of train_df above, so it
    # cannot be used for early stopping or best-model selection without leakage
    # (the model trains on these same images). It is retained ONLY as a
    # monitoring signal in the logs. Model selection is disabled for the final
    # run (select_on_val=False): the epoch budget is fixed (chosen via the
    # 5-fold CV phase) and the LAST-epoch weights are kept as best_model.pth.
    val_df = df[
        (df["source"] == "Shenzhen") & (df["split"] == "val")
    ].reset_index(drop=True)

    logger.info(
        f"Training final U-Net: train+val={len(train_df)} "
        f"(monitoring-only val={len(val_df)}, no val-based selection)"
    )

    _train_single_unet(
        train_df=train_df,
        val_df=val_df,
        cfg=cfg,
        norm_stats=norm_stats,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        experiment_name="unet_final",
        device=device,
        seed=seed,
        select_on_val=False,
        n_epochs_override=cfg["training"].get("final_target_epochs"),
    )

    best_path = os.path.join(checkpoint_dir, "best_model.pth")
    logger.info(f"Final U-Net training complete. Best model: {best_path}")
    return best_path


def _train_single_unet(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: dict,
    norm_stats: dict,
    checkpoint_dir: str,
    log_dir: str,
    experiment_name: str,
    device: torch.device,
    seed: int,
    select_on_val: bool = True,
    n_epochs_override: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Core single-run U-Net training loop.

    Args:
        select_on_val: If True (k-fold CV), `val_df` is a true held-out fold, so
            early stopping and best-model selection use validation Dice. If False
            (final model trained on train+val), `val_df` overlaps the training
            data; selection on it would leak, so it is logged for monitoring only,
            early stopping is disabled, and the LAST epoch is saved as best_model.
        n_epochs_override: If set, train for this many epochs instead of
            cfg['training']['n_epochs'] (the cosine LR schedule still spans the
            configured n_epochs). Used to apply a CV-derived epoch budget to the
            final model.

    Returns:
        (best_val_dice, best_val_iou). When select_on_val is False these are the
        last-epoch metrics (monitoring only).
    """
    from utils.reproducibility import seed_everything
    seed_everything(seed)

    # Build model
    model = build_unet(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        features=cfg["model"]["features"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    # Datasets
    train_transform = get_unet_train_transforms(
        image_size=256,  # unet_size from base config
        norm_stats=norm_stats,
        rotation_limit=cfg["augmentation"]["rotation_limit"],
        brightness_limit=cfg["augmentation"]["brightness_limit"],
        contrast_limit=cfg["augmentation"]["contrast_limit"],
        hflip_prob=cfg["augmentation"]["horizontal_flip_prob"],
        grid_distortion_prob=cfg["augmentation"]["grid_distortion_prob"],
    )
    val_transform = get_unet_val_transforms(image_size=256, norm_stats=norm_stats)

    train_dataset = UNetDataset(
        train_df, transform=train_transform, split_name="train"
    )
    val_dataset = UNetDataset(
        val_df, transform=val_transform, split_name="val"
    )

    worker_init = get_worker_init_fn(seed)

    # PASS-2 fix #8 (twin of the SAM trainer): dedicated generator so batch
    # order is deterministic per (seed, epoch) and resume-stable.
    train_gen = torch.Generator()
    train_gen.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        worker_init_fn=worker_init,
        pin_memory=True,
        drop_last=True,
        generator=train_gen,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["inference"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        worker_init_fn=worker_init,
        pin_memory=True,
    )

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["n_epochs"],
        eta_min=cfg["training"]["scheduler_eta_min"],
    )

    # Loss
    criterion = build_loss(
        loss_name=cfg["training"]["loss"],
        alpha=cfg["training"]["loss_alpha"],
    )

    # AMP scaler (disabled by default for U-Net)
    use_amp = cfg["training"].get("use_amp", False)
    scaler = GradScaler() if use_amp else None

    # Early stopping
    early_stop = EarlyStopping(
        patience=cfg["training"]["early_stopping_patience"],
        min_delta=cfg["training"]["early_stopping_min_delta"],
    )

    # Experiment logger
    exp_logger = ExperimentLogger(log_dir, experiment_name)

    # Resume from checkpoint if available
    latest_ckpt = find_latest_checkpoint(checkpoint_dir, prefix="epoch_")
    start_epoch = 0
    if latest_ckpt:
        ckpt_data = load_checkpoint(
            latest_ckpt, model, optimizer, scheduler, device=device
        )
        start_epoch = ckpt_data.get("epoch", 0) + 1
        logger.info(f"Resuming from epoch {start_epoch}")

    best_val_dice = 0.0
    best_val_iou = 0.0

    # Epoch budget (final model may use a CV-derived override); the cosine LR
    # schedule still spans the configured n_epochs (T_max set above).
    total_epochs = n_epochs_override or cfg["training"]["n_epochs"]

    for epoch in range(start_epoch, total_epochs):
        # PASS-2 fix #8: deterministic, resume-stable batch order + torch ops.
        epoch_seed = seed + epoch
        torch.manual_seed(epoch_seed)
        train_gen.manual_seed(epoch_seed)
        # --- Training phase ---
        model.train()
        train_loss = 0.0
        train_dice_losses = []
        train_bce_losses = []

        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()

            if use_amp and scaler:
                with autocast():
                    logits = model(images)
                    loss, components = criterion(logits, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss, components = criterion(logits, masks)
                loss.backward()
                optimizer.step()

            train_loss += components["total_loss"]
            train_dice_losses.append(components["dice_loss"])
            train_bce_losses.append(components["bce_loss"])

        scheduler.step()

        avg_train_loss = train_loss / len(train_loader)
        avg_dice_loss = np.mean(train_dice_losses)
        avg_bce_loss = np.mean(train_bce_losses)

        # --- Validation phase ---
        val_dice, val_iou = _evaluate_unet(model, val_loader, device)

        exp_logger.log_metrics(
            epoch=epoch,
            split="train",
            total_loss=avg_train_loss,
            dice_loss=avg_dice_loss,
            bce_loss=avg_bce_loss,
            val_dice=val_dice,
            val_iou=val_iou,
            lr=scheduler.get_last_lr()[0],
        )

        if select_on_val:
            # Legitimate held-out val (k-fold): select on validation Dice.
            is_best = val_dice > best_val_dice
            if is_best:
                best_val_dice = val_dice
                best_val_iou = val_iou
        else:
            # Final model: val overlaps train -> no val-based selection.
            # Keep the LAST epoch's weights as best_model.pth.
            best_val_dice = val_dice
            best_val_iou = val_iou
            is_best = (epoch == total_epochs - 1)

        # Save checkpoint every epoch
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics={"dice": val_dice, "iou": val_iou},
            config=cfg,
            checkpoint_dir=checkpoint_dir,
            filename=f"epoch_{epoch:04d}.pth",
            scheduler=scheduler,
            scaler=scaler,
            is_best=is_best,
            # Fix #5: rotate epoch checkpoints to avoid Colab/Drive disk blow-up.
            keep_last_n=cfg["training"].get("keep_last_n_checkpoints", 2),
        )

        logger.info(
            f"Epoch {epoch:3d}/{cfg['training']['n_epochs']} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_dice={val_dice:.4f} | val_iou={val_iou:.4f} | "
            f"best_dice={best_val_dice:.4f}"
        )

        # Early stopping check (on validation Dice) — only when val is a true
        # held-out set. Disabled for the final model (val overlaps train).
        if select_on_val and early_stop.should_stop(val_dice):
            logger.info(
                f"Early stopping at epoch {epoch} "
                f"(no improvement for {early_stop.patience} epochs)"
            )
            break

    exp_logger.close()
    return best_val_dice, best_val_iou


@torch.no_grad()
def _evaluate_unet(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate U-Net on a validation DataLoader.

    Computes per-image Dice and IoU, then returns the mean.

    Audit fix: CRITICAL-18 and LOW-23 — per-image computation.

    Returns:
        (mean_dice, mean_iou) over all images in val_loader.
    """
    model.eval()
    all_dice = []
    all_iou = []

    for batch in val_loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        batch_dice = compute_batch_dice(preds, masks)
        batch_iou = compute_batch_iou(preds, masks)

        all_dice.extend(batch_dice)
        all_iou.extend(batch_iou)

    model.train()
    return float(np.mean(all_dice)), float(np.mean(all_iou))
