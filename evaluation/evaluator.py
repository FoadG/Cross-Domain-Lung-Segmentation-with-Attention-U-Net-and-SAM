"""
evaluation/evaluator.py

Single-domain evaluator: runs inference with any model configuration
on a specified split, computes all metrics, and returns structured results.

Supports all 8 ablation configurations:
    A1: U-Net baseline
    A2: SAM zero-shot (no fine-tuning)
    A3: SAM Freeze-Encoder, single box
    A4: SAM Freeze-Encoder, ensemble
    A5: SAM LoRA, single box
    A6: SAM LoRA, ensemble
    A7: SAM Freeze-Encoder, oracle box (GT)
    A8: SAM Freeze-Encoder, stress-test box

Audit fix: CRITICAL-09 — metrics returned with explicit source labeling
           so cross_domain_eval.py can compute Δ correctly.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import UNetDataset, SAMDataset, sam_collate_fn
from data.transforms import get_unet_val_transforms, resize_mask_pil
from evaluation.metrics import (
    compute_batch_dice,
    compute_batch_iou,
    compute_batch_hd95,
    aggregate_metrics,
)
from prompt_generation.ensemble import (
    run_ensemble_inference,
    run_single_box_inference,
)

logger = logging.getLogger(__name__)


def evaluate_unet(
    model: nn.Module,
    df: pd.DataFrame,
    norm_stats: dict,
    device: torch.device,
    unet_size: int = 256,
    batch_size: int = 16,
    split_name: str = "test",
    source_filter: Optional[str] = None,
) -> Dict:
    """
    Evaluate U-Net on a specified split (Ablation A1).

    Args:
        model: Trained AttentionUNet, in eval mode.
        df: Full DataFrame with split column.
        norm_stats: Pre-computed normalization stats.
        device: Computation device.
        unet_size: U-Net input resolution.
        batch_size: Inference batch size.
        split_name: Split label to evaluate on ('test' or 'cross_domain_test').
        source_filter: If set, only evaluate images from this source.

    Returns:
        Dict with aggregate metrics + per-image lists.
    """
    eval_df = _filter_df(df, split_name, source_filter)
    if len(eval_df) == 0:
        logger.warning(f"No images found for split={split_name}, source={source_filter}")
        return {}

    transform = get_unet_val_transforms(unet_size, norm_stats)
    dataset = UNetDataset(
        eval_df, transform=transform, split_name=split_name,
        assert_no_montgomery_in_training=False,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    model.eval()
    all_dice, all_iou, all_hd95 = [], [], []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            preds = (torch.sigmoid(logits) > 0.5).float()

            all_dice.extend(compute_batch_dice(preds, masks))
            all_iou.extend(compute_batch_iou(preds, masks))
            all_hd95.extend(compute_batch_hd95(preds, masks))

    metrics = aggregate_metrics(all_dice, all_iou, all_hd95)
    metrics["ablation"] = "A1"
    metrics["split"] = split_name
    metrics["source"] = source_filter or "all"
    metrics["dice_per_image"] = all_dice
    metrics["iou_per_image"] = all_iou
    metrics["hd95_per_image"] = all_hd95

    _log_metrics(metrics, "U-Net", split_name, source_filter)
    return metrics


def evaluate_sam(
    finetuner,
    df: pd.DataFrame,
    box_cache_df: pd.DataFrame,
    device: torch.device,
    native_size: int = 512,
    use_ensemble: bool = True,
    batch_size: int = 2,
    split_name: str = "test",
    source_filter: Optional[str] = None,
    ablation_id: str = "A4",
    oracle_box_df: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Evaluate SAM (any fine-tuned variant) on a specified split.

    Supports both single-box and ensemble inference modes.
    For A7 (Oracle), pass oracle_box_df instead of box_cache_df.

    Args:
        finetuner: SAMFineTuner instance (model + processor).
        df: Full DataFrame with split column.
        box_cache_df: Pre-computed pipeline boxes (or oracle boxes for A7).
        device: Computation device.
        native_size: Native pipeline resolution.
        use_ensemble: If True, uses 3-variant ensemble (A4/A6);
                      if False, uses single box (A3/A5).
        batch_size: Number of images per inference call.
        split_name: Split label to evaluate on.
        source_filter: If set, only evaluate this source.
        ablation_id: Label for the ablation (for reporting).
        oracle_box_df: If provided, overrides box_cache_df (for A7).

    Returns:
        Dict with aggregate metrics + per-image lists.
    """
    eval_df = _filter_df(df, split_name, source_filter)
    if len(eval_df) == 0:
        logger.warning(
            f"No images for split={split_name}, source={source_filter}"
        )
        return {}

    # Use oracle boxes if provided (A7)
    active_box_df = oracle_box_df if oracle_box_df is not None else box_cache_df
    box_lookup = {
        row["image_path"]: np.array([
            row["x1_native"], row["y1_native"],
            row["x2_native"], row["y2_native"],
        ], dtype=np.float32)
        for _, row in active_box_df.iterrows()
    }

    from data.transforms import resize_image_pil, resize_mask_pil

    finetuner.model.eval()
    all_dice, all_iou, all_hd95 = [], [], []

    # Process in batches
    image_paths_all = eval_df["image_path"].tolist()
    mask_paths_all = eval_df["mask_path"].tolist()

    for i in range(0, len(image_paths_all), batch_size):
        batch_img_paths = image_paths_all[i: i + batch_size]
        batch_mask_paths = mask_paths_all[i: i + batch_size]

        rgb_images = [
            resize_image_pil(p, native_size, to_rgb=True)
            for p in batch_img_paths
        ]
        gt_masks = torch.stack([
            torch.from_numpy(resize_mask_pil(p, native_size)).unsqueeze(0)
            for p in batch_mask_paths
        ]).to(device)

        boxes = [
            box_lookup.get(
                p,
                np.array([
                    int(0.12 * native_size), int(0.12 * native_size),
                    int(0.88 * native_size), int(0.88 * native_size),
                ], dtype=np.float32)
            )
            for p in batch_img_paths
        ]

        if use_ensemble:
            preds, _ = run_ensemble_inference(
                finetuner=finetuner,
                rgb_images=rgb_images,
                base_boxes=boxes,
                image_paths=batch_img_paths,
                native_size=native_size,
            )
        else:
            preds, _ = run_single_box_inference(
                finetuner=finetuner,
                rgb_images=rgb_images,
                boxes=boxes,
                image_paths=batch_img_paths,
            )

        preds = preds.to(device)
        all_dice.extend(compute_batch_dice(preds, gt_masks))
        all_iou.extend(compute_batch_iou(preds, gt_masks))
        all_hd95.extend(compute_batch_hd95(preds, gt_masks))

    metrics = aggregate_metrics(all_dice, all_iou, all_hd95)
    metrics["ablation"] = ablation_id
    metrics["split"] = split_name
    metrics["source"] = source_filter or "all"
    metrics["use_ensemble"] = use_ensemble
    metrics["dice_per_image"] = all_dice
    metrics["iou_per_image"] = all_iou
    metrics["hd95_per_image"] = all_hd95

    _log_metrics(metrics, f"SAM ({ablation_id})", split_name, source_filter)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _filter_df(
    df: pd.DataFrame,
    split_name: str,
    source_filter: Optional[str],
) -> pd.DataFrame:
    mask = df["split"] == split_name
    if source_filter:
        mask &= df["source"] == source_filter
    return df[mask].reset_index(drop=True)


def _log_metrics(
    metrics: dict,
    model_label: str,
    split_name: str,
    source_filter: Optional[str],
) -> None:
    src = source_filter or "all"
    logger.info(
        f"[Eval] {model_label} | split={split_name} | source={src} | "
        f"Dice={metrics['dice_mean']:.4f}±{metrics['dice_std']:.4f} | "
        f"IoU={metrics['iou_mean']:.4f}±{metrics['iou_std']:.4f} | "
        f"HD95={metrics.get('hd95_mean', float('nan')):.2f} | "
        f"n={metrics['n_images']}"
    )
