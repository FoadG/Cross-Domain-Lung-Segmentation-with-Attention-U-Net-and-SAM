"""
utils/checkpoint.py

Checkpoint management for Colab environments where sessions disconnect.
Every checkpoint includes full metadata for exact resume without data loss.

Audit fix: MODERATE-20 — Colab disconnection protection.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    config: Dict[str, Any],
    checkpoint_dir: str,
    filename: str,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    is_best: bool = False,
    extra: Optional[Dict[str, Any]] = None,
    keep_last_n: Optional[int] = None,
    state_dict_override: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save a complete training checkpoint.

    Saved state includes:
        - Model state dict (or `state_dict_override` if provided)
        - Optimizer state dict
        - Scheduler state dict (if provided)
        - AMP GradScaler state dict (if provided)
        - Current epoch
        - Validation metrics
        - Full config dict
        - Any extra metadata

    Args:
        model: The model to checkpoint.
        optimizer: The optimizer to checkpoint.
        epoch: Current epoch number (0-indexed).
        metrics: Validation metrics dict (e.g., {'dice': 0.91, 'iou': 0.84}).
        config: Full experiment configuration dict.
        checkpoint_dir: Directory to save checkpoints.
        filename: Checkpoint filename (without directory).
        scheduler: Optional LR scheduler to checkpoint.
        scaler: Optional AMP GradScaler to checkpoint.
        is_best: If True, also save a separate 'best_model.pth' copy.
        extra: Optional dict of additional metadata to store.
        keep_last_n: If set, keep only the newest N rotating checkpoints
            (files matching 'epoch_*.pth') in `checkpoint_dir`; older ones are
            deleted after this save. 'best_model.pth' is never pruned. This is
            the fix for audit bug #5 — previously every epoch wrote a full model
            state (~375 MB for SAM) with no rotation, exhausting Colab/Drive
            disk on long runs.
        state_dict_override: If provided, this dict is saved as the model state
            instead of `model.state_dict()`. Used to persist ONLY trainable
            weights (e.g. lora_sam.get_lora_state_dict) so frozen encoder weights
            — reconstructed from the pretrained checkpoint on resume — are not
            duplicated to disk every epoch. Resume must use strict=False.

    Returns:
        Full path to the saved checkpoint file.
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)

    model_state = state_dict_override if state_dict_override is not None else model.state_dict()

    state = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
        # Record whether this is a partial (trainable-only) state so loaders
        # know strict=False is required.
        "partial_state": state_dict_override is not None,
    }

    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()

    if extra is not None:
        state["extra"] = extra

    torch.save(state, checkpoint_path)
    logger.info(
        f"Checkpoint saved: {checkpoint_path} | epoch={epoch} | "
        f"metrics={metrics}"
    )

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(state, best_path)
        logger.info(f"Best model updated: {best_path}")

    if keep_last_n is not None and keep_last_n > 0:
        _prune_old_checkpoints(checkpoint_dir, keep_last_n, prefix="epoch_")

    return checkpoint_path


def _prune_old_checkpoints(
    checkpoint_dir: str, keep_last_n: int, prefix: str = "epoch_"
) -> None:
    """Delete all but the newest `keep_last_n` rotating checkpoints.

    Only files matching '{prefix}*.pth' are considered; 'best_model.pth' and any
    other files are left untouched. Sorting is by the trailing epoch integer so
    that, e.g., epoch_0010 sorts after epoch_0009.
    """
    cdir = Path(checkpoint_dir)
    if not cdir.exists():
        return

    def _epoch_key(p: Path) -> int:
        tail = p.stem.split("_")[-1]
        return int(tail) if tail.isdigit() else -1

    ckpts = sorted(cdir.glob(f"{prefix}*.pth"), key=_epoch_key)
    excess = ckpts[:-keep_last_n] if len(ckpts) > keep_last_n else []
    for old in excess:
        try:
            old.unlink()
            logger.info(f"Pruned old checkpoint: {old}")
        except OSError as e:  # pragma: no cover - defensive
            logger.warning(f"Could not prune {old}: {e}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load a checkpoint and restore model (and optionally optimizer) state.

    Args:
        checkpoint_path: Path to the checkpoint file.
        model: Model to load state into.
        optimizer: Optional optimizer to restore state into.
        scheduler: Optional scheduler to restore state into.
        scaler: Optional AMP GradScaler to restore state into.
        device: Device to map checkpoint tensors to.
        strict: Whether to use strict state dict loading.

    Returns:
        Full checkpoint dictionary (includes 'epoch', 'metrics', etc.).

    Raises:
        FileNotFoundError: If checkpoint_path does not exist.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    metrics = checkpoint.get("metrics", {})
    logger.info(
        f"Checkpoint loaded: {checkpoint_path} | epoch={epoch} | "
        f"metrics={metrics}"
    )

    return checkpoint


def find_latest_checkpoint(checkpoint_dir: str, prefix: str = "epoch_") -> Optional[str]:
    """
    Find the most recent checkpoint in a directory by epoch number.

    Args:
        checkpoint_dir: Directory to search.
        prefix: Filename prefix for checkpoint files.

    Returns:
        Path to the latest checkpoint, or None if directory is empty/missing.
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None

    checkpoints = sorted(
        [
            f for f in checkpoint_path.glob(f"{prefix}*.pth")
        ],
        key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else -1,
    )

    if not checkpoints:
        return None

    latest = str(checkpoints[-1])
    logger.info(f"Found latest checkpoint: {latest}")
    return latest


def save_metrics_json(metrics: Dict[str, Any], output_path: str) -> None:
    """
    Save evaluation metrics to a JSON file for downstream reporting.

    Args:
        metrics: Metrics dictionary (nested or flat).
        output_path: Full path to output JSON file.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to: {output_path}")
