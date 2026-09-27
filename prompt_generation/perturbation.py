"""
prompt_generation/perturbation.pyش

Box perturbation for SAM training robustness (PP-SAM style).

During fine-tuning, each box prompt is randomly perturbed so the model
learns to produce accurate masks even with slightly inaccurate prompts.
This is critical because the pipeline U-Net will never produce a perfect box.

Reference:
    Rahman et al., "PP-SAM: Perturbed Prompts for Robust Adaptation of
    Segment Anything Model for Polyp Segmentation", 2024.
    "bounding boxes are randomly expanded between 0 and 50 pixels"

Audit fix: CRITICAL-04 (prompt robustness), MODERATE-21 (seed control).
"""

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def perturb_box(
    box: np.ndarray,
    min_pixels: int = 0,
    max_pixels: int = 30,
    image_size: int = 512,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """
    Apply random perturbation to a single bounding box.

    Perturbation is applied independently to each of the four coordinates.
    Box boundaries are clipped to remain within the image.

    The magnitude range [min_pixels, max_pixels] is relative to native_size.
    Default 0–30 pixels (on 512×512) ≈ 0–5.9% of image dimension,
    similar to PP-SAM's 0–50px on 1024×1024 images.

    Args:
        box: np.ndarray [x1, y1, x2, y2] in image_size pixel coords.
        min_pixels: Minimum perturbation magnitude.
        max_pixels: Maximum perturbation magnitude.
        image_size: Image resolution for clipping.
        rng: Optional numpy RandomState for reproducibility.
             If None, falls back to the GLOBAL np.random generator, which is
             seeded by utils.reproducibility.seed_everything. (The previous
             behaviour constructed a fresh, OS-seeded RandomState() here, which
             silently bypassed the global seed and made SAM training
             non-reproducible — audit bug #8.)

    Returns:
        Perturbed box [x1, y1, x2, y2], clipped to [0, image_size-1].
    """
    # Fix #8: when no explicit RNG is supplied, use the seeded global generator
    # (np.random) rather than a fresh unseeded RandomState().
    gen = rng if rng is not None else np.random

    noise = gen.randint(min_pixels, max_pixels + 1, size=4).astype(np.float32)

    # Apply random sign independently for each coordinate
    signs = gen.choice([-1, 1], size=4)
    noise = noise * signs

    perturbed = box + noise

    # Clip to image boundaries
    perturbed[0] = np.clip(perturbed[0], 0, image_size - 1)  # x1
    perturbed[1] = np.clip(perturbed[1], 0, image_size - 1)  # y1
    perturbed[2] = np.clip(perturbed[2], 0, image_size - 1)  # x2
    perturbed[3] = np.clip(perturbed[3], 0, image_size - 1)  # y2

    # Ensure x1 < x2 and y1 < y2 (even after perturbation)
    if perturbed[0] >= perturbed[2]:
        perturbed[0] = max(0, perturbed[2] - 1)
    if perturbed[1] >= perturbed[3]:
        perturbed[1] = max(0, perturbed[3] - 1)

    return perturbed.astype(np.float32)


def perturb_box_batch(
    boxes: List[np.ndarray],
    min_pixels: int = 0,
    max_pixels: int = 30,
    image_size: int = 512,
    rng: Optional[np.random.RandomState] = None,
) -> List[np.ndarray]:
    """
    Apply perturbation to a list of boxes (one per image in a batch).

    Each box gets independent random perturbation.

    Args:
        boxes: List of [x1, y1, x2, y2] arrays in image_size pixel coords.
        min_pixels: Minimum perturbation magnitude.
        max_pixels: Maximum perturbation magnitude.
        image_size: Image resolution for clipping.
        rng: Optional seeded RandomState for reproducibility. Passing a per-epoch
            seeded RNG (see sam_trainer) makes perturbations deterministic and
            resume-stable. If None, the seeded global np.random is used.

    Returns:
        List of perturbed boxes, same length and format as input.
    """
    perturbed_boxes = []
    for box in boxes:
        perturbed_boxes.append(
            perturb_box(box, min_pixels, max_pixels, image_size, rng=rng)
        )
    return perturbed_boxes


def create_ensemble_box_variants(
    base_box: np.ndarray,
    image_size: int = 512,
    extra_margin_fraction: float = 0.05,
    shift_fraction: float = 0.05,
) -> List[np.ndarray]:
    """
    Create three box variants for ensemble inference (SAM-U style).

    Variant 1: Original box (base_box, no change).
    Variant 2: Larger box (extra margin on all sides).
    Variant 3: Shifted box (small shift in both directions).

    These three variants are deterministic (not random) — they produce
    identical results on the same image every time, ensuring ensemble
    reproducibility during evaluation.

    Reference:
        Deng et al., "SAM-U: Multi-box prompts triggered uncertainty
        estimation for reliable SAM in medical image", 2023.

    Args:
        base_box: [x1, y1, x2, y2] in image_size pixel coords.
        image_size: Coordinate space for clipping.
        extra_margin_fraction: Fraction of image_size for extra expansion
                               in variant 2.
        shift_fraction: Fraction of box dimensions for shift in variant 3.

    Returns:
        List of 3 boxes: [original, larger, shifted].
    """
    x1, y1, x2, y2 = base_box

    # Variant 1: no change
    v1 = base_box.copy()

    # Variant 2: uniformly larger
    extra_px = extra_margin_fraction * image_size
    v2 = np.array([
        max(0, x1 - extra_px),
        max(0, y1 - extra_px),
        min(image_size - 1, x2 + extra_px),
        min(image_size - 1, y2 + extra_px),
    ], dtype=np.float32)

    # Variant 3: shifted right and down (deterministic offset)
    box_w = x2 - x1
    box_h = y2 - y1
    shift_x = shift_fraction * box_w
    shift_y = shift_fraction * box_h
    v3 = np.array([
        min(image_size - 1, x1 + shift_x),
        min(image_size - 1, y1 + shift_y),
        min(image_size - 1, x2 + shift_x),
        min(image_size - 1, y2 + shift_y),
    ], dtype=np.float32)

    return [v1, v2, v3]
