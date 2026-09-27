"""
evaluation/metrics.py

Segmentation evaluation metrics for binary lung masks.

All metrics are computed per-image first, then averaged.
This matches clinical evaluation standards and avoids bias from
images with different mask sizes (audit fix LOW-23).

Metrics implemented:
  - Dice Coefficient (F1 for binary segmentation)
  - IoU / Jaccard Index
  - HD95 (Hausdorff Distance, 95th percentile) — scipy only, no medpy

Audit fix: CRITICAL-08 — HD95 via scipy.spatial.cKDTree, no medpy.
Audit fix: LOW-23 — per-image computation before batch averaging.
"""

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion

logger = logging.getLogger(__name__)

# Last-resort fallback penalty if image dimensions are unavailable. By default
# compute_hd95 now derives a RESOLUTION-AWARE penalty (the image diagonal), so
# this constant is only used when a caller cannot supply mask shape (audit #7).
EMPTY_MASK_HD95_PENALTY = 512.0
SMOOTH = 1e-6


def compute_dice(
    pred: np.ndarray,
    gt: np.ndarray,
    smooth: float = SMOOTH,
) -> float:
    """
    Compute Dice coefficient for a single image.

    Args:
        pred: Binary prediction mask (H, W), values in {0, 1}.
        gt:   Binary ground-truth mask (H, W), values in {0, 1}.
        smooth: Smoothing constant to prevent 0/0.

    Returns:
        Dice score in [0, 1].
    """
    pred_flat = pred.astype(np.float64).ravel()
    gt_flat = gt.astype(np.float64).ravel()
    intersection = (pred_flat * gt_flat).sum()
    return float(
        (2.0 * intersection + smooth) / (pred_flat.sum() + gt_flat.sum() + smooth)
    )


def compute_iou(
    pred: np.ndarray,
    gt: np.ndarray,
    smooth: float = SMOOTH,
) -> float:
    """
    Compute Intersection-over-Union (Jaccard Index) for a single image.

    Args:
        pred: Binary prediction mask (H, W).
        gt:   Binary ground-truth mask (H, W).
        smooth: Smoothing constant.

    Returns:
        IoU score in [0, 1].
    """
    pred_flat = pred.astype(np.float64).ravel()
    gt_flat = gt.astype(np.float64).ravel()
    intersection = (pred_flat * gt_flat).sum()
    union = pred_flat.sum() + gt_flat.sum() - intersection
    return float((intersection + smooth) / (union + smooth))


def _surface_points(mask_bool: np.ndarray) -> np.ndarray:
    """Return (N, 2) coordinates of a binary mask's boundary (surface) pixels.

    Boundary = mask AND NOT erosion(mask), with border_value=0 so that pixels
    on the image edge are correctly treated as surface. Hausdorff distance is
    defined over the object SURFACE, not its filled interior.
    """
    eroded = binary_erosion(mask_bool, border_value=0)
    return np.argwhere(mask_bool & ~eroded)


def compute_hd95(
    pred: np.ndarray,
    gt: np.ndarray,
    percentile: float = 95.0,
    voxel_spacing: float = 1.0,
    empty_penalty: Optional[float] = None,
) -> float:
    """
    Compute the 95th-percentile (symmetric) Hausdorff Distance.

    Uses SURFACE (boundary) points and a pooled directed-distance percentile,
    matching the de-facto reference implementation (medpy.metric.binary.hd95);
    verified to agree with medpy to within 1e-6 on test masks.

    Audit fix #7: the previous version measured distances between ALL foreground
    pixels rather than boundary pixels. Because overlapping interior pixels have
    nearest-neighbour distance 0, that pooled distribution was dominated by
    zeros and systematically UNDER-reported boundary disagreement (e.g. 2.8 vs a
    true 8.0 px on a shifted disk), making HD95 incomparable to the literature.

    Args:
        pred: Binary prediction mask (H, W), values in {0, 1}.
        gt:   Binary ground-truth mask (H, W), values in {0, 1}.
        percentile: Distance percentile for robust HD (95 by default).
        voxel_spacing: Physical spacing per pixel (use 1.0 for pixel units).
        empty_penalty: Distance returned when exactly one mask is empty. If None
            (default), a RESOLUTION-AWARE penalty equal to the image diagonal
            (sqrt(H^2 + W^2) * voxel_spacing) is used instead of a hard-coded
            512, so the metric is meaningful at any resolution.

    Returns:
        HD95 in pixels (or voxel_spacing units). Returns 0.0 when BOTH masks are
        empty (perfect agreement) and `empty_penalty` when exactly one is empty.
    """
    pred_b = pred > 0.5
    gt_b = gt > 0.5

    if empty_penalty is None:
        h, w = pred_b.shape[-2], pred_b.shape[-1]
        empty_penalty = float(np.hypot(h, w)) * voxel_spacing

    pred_sum = int(pred_b.sum())
    gt_sum = int(gt_b.sum())

    if pred_sum == 0 and gt_sum == 0:
        # Both empty -> no boundary disagreement.
        return 0.0
    if pred_sum == 0 or gt_sum == 0:
        logger.debug(
            "HD95: exactly one empty mask "
            f"(pred={pred_sum}, gt={gt_sum}); returning penalty {empty_penalty:.2f}."
        )
        return float(empty_penalty)

    pred_pts = _surface_points(pred_b) * voxel_spacing
    gt_pts = _surface_points(gt_b) * voxel_spacing

    pred_tree = cKDTree(pred_pts)
    gt_tree = cKDTree(gt_pts)

    # Directed surface distances in both directions.
    d_pred_to_gt, _ = gt_tree.query(pred_pts)
    d_gt_to_pred, _ = pred_tree.query(gt_pts)

    # Pooled symmetric HD at the requested percentile (medpy-equivalent).
    all_distances = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(all_distances, percentile))


# ─────────────────────────────────────────────────────────────────────────────
# Batch-level helpers (called during training and evaluation)
# ─────────────────────────────────────────────────────────────────────────────

def compute_batch_dice(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> List[float]:
    """
    Compute per-image Dice for a batch of predictions.

    Args:
        preds:   (B, 1, H, W) binary predictions (float32, values in {0, 1}).
        targets: (B, 1, H, W) binary ground truth (float32, values in {0, 1}).

    Returns:
        List of Dice scores, one per image in the batch.
    """
    B = preds.shape[0]
    scores = []
    for i in range(B):
        p = preds[i, 0].cpu().numpy()
        t = targets[i, 0].cpu().numpy()
        scores.append(compute_dice(p, t))
    return scores


def compute_batch_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> List[float]:
    """
    Compute per-image IoU for a batch of predictions.

    Args:
        preds:   (B, 1, H, W) binary predictions.
        targets: (B, 1, H, W) binary ground truth.

    Returns:
        List of IoU scores, one per image in the batch.
    """
    B = preds.shape[0]
    scores = []
    for i in range(B):
        p = preds[i, 0].cpu().numpy()
        t = targets[i, 0].cpu().numpy()
        scores.append(compute_iou(p, t))
    return scores


def compute_batch_hd95(
    preds: torch.Tensor,
    targets: torch.Tensor,
    percentile: float = 95.0,
) -> List[float]:
    """
    Compute per-image HD95 for a batch.

    Args:
        preds:   (B, 1, H, W) binary predictions.
        targets: (B, 1, H, W) binary ground truth.
        percentile: HD percentile (default 95).

    Returns:
        List of HD95 values, one per image.
    """
    B = preds.shape[0]
    scores = []
    for i in range(B):
        p = preds[i, 0].cpu().numpy()
        t = targets[i, 0].cpu().numpy()
        scores.append(compute_hd95(p, t, percentile=percentile))
    return scores


def aggregate_metrics(
    dice_list: List[float],
    iou_list: List[float],
    hd95_list: Optional[List[float]] = None,
) -> dict:
    """
    Compute mean ± std for all metric lists.

    Args:
        dice_list: Per-image Dice scores.
        iou_list:  Per-image IoU scores.
        hd95_list: Per-image HD95 values (optional).

    Returns:
        Dict with keys: dice_mean, dice_std, iou_mean, iou_std,
        and optionally hd95_mean, hd95_std.
    """
    result = {
        "dice_mean": float(np.mean(dice_list)),
        "dice_std":  float(np.std(dice_list)),
        "iou_mean":  float(np.mean(iou_list)),
        "iou_std":   float(np.std(iou_list)),
        "n_images":  len(dice_list),
    }
    if hd95_list is not None:
        result["hd95_mean"] = float(np.mean(hd95_list))
        result["hd95_std"]  = float(np.std(hd95_list))
    return result
