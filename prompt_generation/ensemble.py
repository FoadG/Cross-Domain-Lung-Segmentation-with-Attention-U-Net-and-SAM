"""
prompt_generation/ensemble.py

Ensemble inference for SAM using multiple box variants.

Strategy (SAM-U style):
    1. For each image, create 3 deterministic box variants.
    2. Run SAM independently with each variant.
    3. Combine via per-pixel majority voting (threshold ≥ vote_threshold).
    4. Detect and log empty ensemble outputs (audit fix CRITICAL-15).

This reduces sensitivity to the exact box position and size,
which is critical because U-Net boxes are imperfect.

Audit fix: CRITICAL-15 — empty ensemble output detection and logging.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from prompt_generation.perturbation import create_ensemble_box_variants

logger = logging.getLogger(__name__)


def run_ensemble_inference(
    finetuner,
    rgb_images: List[np.ndarray],
    base_boxes: List[np.ndarray],
    image_paths: Optional[List[str]] = None,
    native_size: int = 512,
    vote_threshold: float = 0.5,
    extra_margin_fraction: float = 0.05,
    shift_fraction: float = 0.05,
) -> Tuple[torch.Tensor, Dict]:
    """
    Run deterministic ensemble inference using 3 box variants per image.

    For each image:
        - Creates 3 box variants (original, larger, shifted)
        - Gets SAM prediction for each variant
        - Combines by majority voting (pixel must be predicted positive
          in ≥ vote_threshold fraction of variants)
        - Detects and logs empty final masks

    Audit fix: CRITICAL-15 — empty mask detection and logging.

    Args:
        finetuner: SAMFineTuner instance (model + processor).
        rgb_images: List of (H, W, 3) uint8 numpy arrays.
        base_boxes: List of [x1, y1, x2, y2] arrays in native_size pixel coords.
        image_paths: Optional list of image file paths for cache lookup.
        native_size: Resolution of images and boxes.
        vote_threshold: Minimum fraction of positive votes for a pixel to be 1.
        extra_margin_fraction: Expansion for larger box variant.
        shift_fraction: Shift fraction for shifted box variant.

    Returns:
        ensemble_masks: (B, 1, H, W) binary float32 tensor.
        info: Dict with:
            - 'vote_maps': (B, 1, H, W) float tensors with raw vote fractions
            - 'empty_indices': List of batch indices with empty final masks
            - 'n_empty': Number of images with empty ensemble output
    """
    B = len(rgb_images)
    device = next(finetuner.model.parameters()).device

    # Create 3 variants for each image
    # Shape: [n_variants][B] → n_variants lists of B boxes
    n_variants = 3
    variant_boxes_per_image = [
        create_ensemble_box_variants(
            base_box=base_boxes[i],
            image_size=native_size,
            extra_margin_fraction=extra_margin_fraction,
            shift_fraction=shift_fraction,
        )
        for i in range(B)
    ]  # variant_boxes_per_image[i][v] = box for image i, variant v

    # Transpose: variant_boxes[v][i] = box for variant v, image i
    variant_boxes = [
        [variant_boxes_per_image[i][v] for i in range(B)]
        for v in range(n_variants)
    ]

    # Run SAM for each variant
    variant_masks = []
    for v in range(n_variants):
        boxes_v = variant_boxes[v]

        with torch.no_grad():
            pred_masks_v, _ = finetuner.forward_with_box(
                rgb_images=rgb_images,
                boxes=boxes_v,
                image_paths=image_paths,
            )
        # pred_masks_v: (B, 1, H, W) RAW LOGITS — apply sigmoid before threshold

        binary_v = (torch.sigmoid(pred_masks_v) > 0.5).float()
        variant_masks.append(binary_v)

    # Stack: (n_variants, B, 1, H, W)
    stacked = torch.stack(variant_masks, dim=0)

    # Vote map: mean across variants (B, 1, H, W)
    vote_map = stacked.mean(dim=0)

    # Final ensemble mask
    ensemble_masks = (vote_map >= vote_threshold).float()

    # Detect empty outputs (audit fix CRITICAL-15)
    empty_indices = []
    for i in range(B):
        if ensemble_masks[i].sum() == 0:
            empty_indices.append(i)
            path_info = image_paths[i] if image_paths else f"image_{i}"
            logger.warning(
                f"EMPTY ENSEMBLE MASK at index {i} ({path_info}). "
                f"Vote map max: {vote_map[i].max().item():.3f}. "
                "Mask will be all-zeros for this image."
            )

    if empty_indices:
        logger.warning(
            f"Ensemble produced {len(empty_indices)}/{B} empty masks. "
            "This may indicate poor box quality or domain mismatch."
        )

    info = {
        "vote_maps": vote_map,
        "empty_indices": empty_indices,
        "n_empty": len(empty_indices),
        "n_variants": n_variants,
    }

    return ensemble_masks, info


def run_single_box_inference(
    finetuner,
    rgb_images: List[np.ndarray],
    boxes: List[np.ndarray],
    image_paths: Optional[List[str]] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    Run SAM with a single box per image (no ensemble).

    Used for ablation A3 (Freeze-Encoder, single box baseline).

    Args:
        finetuner: SAMFineTuner instance.
        rgb_images: List of (H, W, 3) uint8 numpy arrays.
        boxes: List of [x1, y1, x2, y2] in native_size pixel coords.
        image_paths: Optional paths for cache lookup.

    Returns:
        masks: (B, 1, H, W) binary float32 tensor.
        info: Dict with basic statistics.
    """
    with torch.no_grad():
        pred_masks, raw_outputs = finetuner.forward_with_box(
            rgb_images=rgb_images,
            boxes=boxes,
            image_paths=image_paths,
        )

    binary_masks = (torch.sigmoid(pred_masks) > 0.5).float()  # logits → binary
    empty_count = sum(
        1 for i in range(len(rgb_images)) if binary_masks[i].sum() == 0
    )

    info = {
        "n_empty": empty_count,
        "raw_outputs": raw_outputs,
    }
    return binary_masks, info
