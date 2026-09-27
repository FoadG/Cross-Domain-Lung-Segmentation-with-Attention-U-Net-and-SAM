"""
prompt_generation/box_extractor.py

Extracts bounding-box prompts from binary segmentation masks.

Coordinate chain (audit fix CRITICAL-01):
    U-Net mask (256×256)
        → box in 256-space
        → scale to native_size (512) space
        → passed to SamProcessor as native_size pixel coords
        → SamProcessor internally scales to 1024×1024

All three box variants needed for the ablation study:
    - Pipeline box:    from U-Net prediction (normal inference path)
    - Oracle box (A7): from ground-truth mask (upper-bound experiment)
    - Stress-test box (A8): deliberately corrupted for sensitivity analysis

Audit fixes:
    CRITICAL-01 — explicit coordinate scaling at every step
    CRITICAL-07 — fallback box loaded from pre-computed JSON
    CRITICAL-10 — oracle path is completely separate function

    *** NEW — CRITICAL bug #3 (silent train/serve skew) ***
    generate_prompt_cache() previously normalized inference images with a bare
    `img / 255.0`, but the U-Net was trained AND validated with
    A.Normalize(mean, std) (see data/transforms.get_unet_val_transforms). The
    model therefore saw inputs in [0, 1] at prompt-generation time vs roughly
    [-2.2, 1.8] during training — a large distribution shift that degraded every
    PREDICTED prompt box, propagating into all SAM Freeze/LoRA pipeline
    ablations (A3/A4/A5/A6 and the RQs built on them).

    Fix: prompt generation now applies the EXACT same val transform the U-Net
    was trained against, reusing get_unet_val_transforms(unet_size, norm_stats).
    The function now requires the normalization stats (dict or JSON path) and
    raises a clear error if they are missing, so the skew can never recur
    silently. The previously-imported-but-unused get_unet_val_transforms is now
    actually used; the unused torch.nn.functional import was removed.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import label as scipy_label

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core box extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_box_from_mask(
    mask: np.ndarray,
    image_size: int,
    margin_fraction: float = 0.12,
    min_component_area_fraction: float = 0.01,
) -> Tuple[Optional[np.ndarray], bool]:
    """
    Extract an outer bounding box from a binary mask.

    Steps:
        1. Label connected components.
        2. Keep the largest component with area > min_component_area_fraction.
        3. Compute tight bounding box of that component.
        4. Expand by margin_fraction (CRITICAL: must be outer-box, not inner-box).
        5. Clip to image boundaries.

    Audit fix: CRITICAL-01, 10 — outer-box guarantee.

    Args:
        mask: Binary float32 array (H, W), values in {0.0, 1.0}.
              Assumed to be at U-Net output resolution (e.g. 256×256).
        image_size: The resolution of this mask (e.g. 256 for U-Net output).
        margin_fraction: Fraction of image_size to add to each side.
        min_component_area_fraction: Minimum fraction of total pixels a
            component must occupy to be considered a valid lung region.

    Returns:
        box: np.ndarray [x1, y1, x2, y2] in image_size pixel coords,
             or None if no valid component found.
        used_fallback: True if fallback was triggered (no valid component).
    """
    binary = (mask > 0.5).astype(np.int32)

    if binary.sum() == 0:
        return None, True

    labeled, n_components = scipy_label(binary)

    if n_components == 0:
        return None, True

    # Find the largest component
    min_area = int(min_component_area_fraction * image_size * image_size)
    component_areas = [
        (i + 1, (labeled == i + 1).sum())
        for i in range(n_components)
    ]
    valid_components = [
        (idx, area) for idx, area in component_areas if area >= min_area
    ]

    if not valid_components:
        return None, True

    # Largest valid component
    best_idx = max(valid_components, key=lambda x: x[1])[0]
    component_mask = (labeled == best_idx)

    # Tight bounding box (row=y, col=x)
    rows = np.any(component_mask, axis=1)
    cols = np.any(component_mask, axis=0)

    y1 = int(np.argmax(rows))
    y2 = int(image_size - np.argmax(rows[::-1]) - 1)
    x1 = int(np.argmax(cols))
    x2 = int(image_size - np.argmax(cols[::-1]) - 1)

    # Apply outer-box margin (audit fix: must add, not subtract)
    margin_px = int(margin_fraction * image_size)
    x1 = max(0, x1 - margin_px)
    y1 = max(0, y1 - margin_px)
    x2 = min(image_size - 1, x2 + margin_px)
    y2 = min(image_size - 1, y2 + margin_px)

    box = np.array([x1, y1, x2, y2], dtype=np.float32)
    return box, False


def scale_box(
    box: np.ndarray,
    from_size: int,
    to_size: int,
) -> np.ndarray:
    """
    Scale a bounding box from one resolution to another.

    Audit fix: CRITICAL-01 — explicit coordinate system scaling.

    Args:
        box: [x1, y1, x2, y2] in from_size pixel coords.
        from_size: Source resolution (e.g. 256 for U-Net output).
        to_size: Target resolution (e.g. 512 for native pipeline size).

    Returns:
        Scaled [x1, y1, x2, y2] in to_size pixel coords.
    """
    scale = to_size / from_size
    scaled = box * scale
    # Clip after scaling to handle floating-point overshoot
    scaled = np.clip(scaled, 0, to_size - 1)
    return scaled.astype(np.float32)


def get_oracle_box(
    gt_mask: np.ndarray,
    unet_size: int = 256,
    native_size: int = 512,
    margin_fraction: float = 0.12,
    min_component_area_fraction: float = 0.01,
) -> Tuple[np.ndarray, bool]:
    """
    Extract box from the GROUND-TRUTH mask. Used for Ablation A7 (Oracle Bound).

    This is a completely separate code path from get_pipeline_box().
    Calling the wrong function accidentally is prevented by distinct naming.

    Audit fix: CRITICAL-10 — oracle path is explicitly separated.

    Args:
        gt_mask: Binary float32 ground-truth mask at unet_size resolution.
        unet_size: Resolution of gt_mask.
        native_size: Target resolution for the returned box.
        margin_fraction: Box expansion margin.
        min_component_area_fraction: Minimum component area threshold.

    Returns:
        box: [x1, y1, x2, y2] in native_size pixel coords.
        used_fallback: Whether the fallback was needed.
    """
    box_256, used_fallback = extract_box_from_mask(
        mask=gt_mask,
        image_size=unet_size,
        margin_fraction=margin_fraction,
        min_component_area_fraction=min_component_area_fraction,
    )

    if box_256 is None:
        # Fallback: full image box
        m = int(margin_fraction * native_size)
        return np.array([m, m, native_size - m, native_size - m], np.float32), True

    box_native = scale_box(box_256, from_size=unet_size, to_size=native_size)
    return box_native, used_fallback


def get_pipeline_box(
    pred_mask: np.ndarray,
    fallback_box_stats: dict,
    unet_size: int = 256,
    native_size: int = 512,
    margin_fraction: float = 0.12,
    min_component_area_fraction: float = 0.01,
) -> Tuple[np.ndarray, bool]:
    """
    Extract box from U-Net PREDICTED mask. Normal inference path (A3/A4/A5/A6).

    If U-Net prediction is empty or below area threshold, the pre-computed
    fallback box is used (audit fix CRITICAL-07).

    Audit fix: CRITICAL-10 — pipeline path is explicitly named.

    Args:
        pred_mask: Binary float32 U-Net prediction at unet_size resolution.
        fallback_box_stats: Dict from cleaner.compute_fallback_box_stats().
            Keys: x1_norm, y1_norm, x2_norm, y2_norm (normalized 0–1).
        unet_size: U-Net output resolution.
        native_size: Target resolution for returned box.
        margin_fraction: Box expansion margin.
        min_component_area_fraction: Minimum component area threshold.

    Returns:
        box: [x1, y1, x2, y2] in native_size pixel coords.
        used_fallback: True if statistical fallback was used.
    """
    box_256, used_fallback = extract_box_from_mask(
        mask=pred_mask,
        image_size=unet_size,
        margin_fraction=margin_fraction,
        min_component_area_fraction=min_component_area_fraction,
    )

    if box_256 is None:
        # Use pre-computed statistical fallback (stored in normalized coords)
        box_native = _normalized_to_pixel(fallback_box_stats, native_size)
        logger.debug(
            "Fallback box used (U-Net produced no valid component). "
            f"Box: {box_native}"
        )
        return box_native, True

    box_native = scale_box(box_256, from_size=unet_size, to_size=native_size)
    return box_native, False


def get_stress_test_box(
    gt_mask: np.ndarray,
    error_scale: float = 0.25,
    unet_size: int = 256,
    native_size: int = 512,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a deliberately corrupted box for Ablation A8 (stress test).

    Applies large random perturbation to the oracle box to test
    SAM's sensitivity to prompt errors beyond the training perturbation range.

    Audit fix: CRITICAL-10 — stress test path explicitly named.

    Args:
        gt_mask: Ground-truth mask (used as base for oracle box).
        error_scale: Perturbation scale as fraction of native_size.
                     0.25 = up to ±25% position error (large, stressful).
        unet_size: GT mask resolution.
        native_size: Target box resolution.
        seed: For reproducible stress test.

    Returns:
        Corrupted [x1, y1, x2, y2] in native_size pixel coords.
    """
    rng = np.random.RandomState(seed)
    oracle_box, _ = get_oracle_box(gt_mask, unet_size, native_size)

    # Add large random perturbations
    max_error = int(error_scale * native_size)
    noise = rng.randint(-max_error, max_error, size=4).astype(np.float32)
    corrupted = oracle_box + noise
    corrupted = np.clip(corrupted, 0, native_size - 1)

    # Ensure x1 < x2 and y1 < y2
    corrupted[0] = min(corrupted[0], corrupted[2] - 1)
    corrupted[1] = min(corrupted[1], corrupted[3] - 1)

    return corrupted.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Batch processing: generate boxes for all images and save to CSV
# ─────────────────────────────────────────────────────────────────────────────

def generate_prompt_cache(
    df: pd.DataFrame,
    unet_model,
    fallback_box_stats: dict,
    device,
    unet_size: int = 256,
    native_size: int = 512,
    margin_fraction: float = 0.12,
    min_component_area_fraction: float = 0.01,
    output_csv_path: str = "./data/processed/prompt_boxes.csv",
    batch_size: int = 16,
    norm_stats: Optional[dict] = None,
    norm_stats_json_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run U-Net inference on all images and save bounding boxes to CSV.

    This runs ONCE after final U-Net training and produces the prompt CSV
    that SAMDataset reads during SAM training and evaluation.

    Args:
        df: Full DataFrame with all image records.
        unet_model: Trained final U-Net (loaded, on device).
        fallback_box_stats: Fallback box statistics from cleaner.py.
        device: Computation device.
        unet_size: U-Net input/output resolution.
        native_size: Native pipeline resolution for output box coords.
        margin_fraction: Box expansion margin.
        min_component_area_fraction: Minimum valid component area fraction.
        output_csv_path: Where to save the box CSV.
        batch_size: Batch size for U-Net inference.
        norm_stats: U-Net normalization stats dict ({'mean': [...], 'std': [...]})
            as produced by cleaner.compute_normalization_stats. REQUIRED (either
            this or norm_stats_json_path) — without it the U-Net is fed
            mis-normalized inputs (the original train/serve-skew bug #3).
        norm_stats_json_path: Path to norm_stats.json; used to load norm_stats
            when the dict is not passed directly.

    Returns:
        DataFrame with columns:
            image_path, x1_native, y1_native, x2_native, y2_native,
            used_fallback, source, split
    """
    import torch
    from data.transforms import get_unet_val_transforms, load_norm_stats
    from PIL import Image

    # --- Fix #3: resolve the SAME normalization the U-Net was trained with ---
    if norm_stats is None and norm_stats_json_path is not None:
        norm_stats = load_norm_stats(norm_stats_json_path)
    if norm_stats is None:
        raise ValueError(
            "generate_prompt_cache requires the U-Net normalization stats "
            "(mean/std). Pass norm_stats=<dict> or "
            "norm_stats_json_path=paths['norm_stats']. Without them the U-Net "
            "would receive inputs in [0,1] (bare /255) while it was trained on "
            "A.Normalize(mean,std) inputs — a silent train/serve skew that "
            "degrades every predicted prompt box (audit bug #3)."
        )

    # Identical pipeline to validation/inference: Resize -> Normalize -> ToTensor.
    val_transform = get_unet_val_transforms(unet_size, norm_stats)

    logger.info(
        f"Generating prompt boxes for {len(df)} images "
        f"(norm: mean={norm_stats['mean'][0]:.4f}, std={norm_stats['std'][0]:.4f})..."
    )

    unet_model.eval()
    rows = []
    fallback_count = 0

    # Process per image (not batched, to handle variable original sizes)
    for i, (_, row) in enumerate(df.iterrows()):
        image_path = row["image_path"]

        # Load full-resolution grayscale; the val transform handles resize +
        # normalization exactly as during training (no more bare /255 skew).
        with Image.open(image_path) as img:
            gray = np.array(img.convert("L"))  # uint8 (H, W), original resolution

        transformed = val_transform(image=gray)
        # ToTensorV2 on 2D grayscale -> (1, H, W) float32; add batch dim -> (1,1,H,W)
        img_tensor = transformed["image"].unsqueeze(0).to(device).float()

        with torch.no_grad():
            logit = unet_model(img_tensor)
            prob = torch.sigmoid(logit)[0, 0].cpu().numpy()

        pred_mask = (prob > 0.5).astype(np.float32)

        box_native, used_fallback = get_pipeline_box(
            pred_mask=pred_mask,
            fallback_box_stats=fallback_box_stats,
            unet_size=unet_size,
            native_size=native_size,
            margin_fraction=margin_fraction,
            min_component_area_fraction=min_component_area_fraction,
        )

        if used_fallback:
            fallback_count += 1

        rows.append({
            "image_path": image_path,
            "x1_native": float(box_native[0]),
            "y1_native": float(box_native[1]),
            "x2_native": float(box_native[2]),
            "y2_native": float(box_native[3]),
            "used_fallback": used_fallback,
            "source": row["source"],
            "split": row["split"],
        })

        if (i + 1) % 50 == 0:
            logger.info(f"  Prompt generation: {i + 1}/{len(df)} images")

    box_df = pd.DataFrame(rows)
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    box_df.to_csv(output_csv_path, index=False)

    fallback_rate = fallback_count / len(df) * 100
    logger.info(
        f"Prompt cache saved: {output_csv_path} | "
        f"fallback_rate={fallback_rate:.1f}% ({fallback_count}/{len(df)})"
    )

    if fallback_rate > 10.0:
        logger.warning(
            f"HIGH FALLBACK RATE ({fallback_rate:.1f}%) — "
            "U-Net quality may be insufficient for reliable prompt generation. "
            "Consider re-training with better hyperparameters."
        )

    return box_df


# ─────────────────────────────────────────────────────────────────────────────
# Oracle box cache (for Ablation A7)
# ─────────────────────────────────────────────────────────────────────────────

def generate_oracle_prompt_cache(
    df: pd.DataFrame,
    unet_size: int = 256,
    native_size: int = 512,
    margin_fraction: float = 0.12,
    output_csv_path: str = "./data/processed/oracle_boxes.csv",
) -> pd.DataFrame:
    """
    Generate oracle bounding boxes from ground-truth masks (Ablation A7).

    Completely separate from generate_prompt_cache() to prevent accidental
    use of GT boxes in the live pipeline.

    Audit fix: CRITICAL-10 — oracle path is clearly isolated.

    Args:
        df: Full DataFrame with mask_path column.
        unet_size: Resolution to resize GT masks to for box extraction.
        native_size: Target box coordinate space.
        margin_fraction: Box expansion margin.
        output_csv_path: Where to save oracle box CSV.

    Returns:
        DataFrame with oracle box coordinates.
    """
    from data.transforms import resize_mask_pil

    logger.info(f"Generating ORACLE boxes for {len(df)} images (Ablation A7)...")
    rows = []

    for _, row in df.iterrows():
        gt_mask = resize_mask_pil(row["mask_path"], unet_size)
        box_native, used_fallback = get_oracle_box(
            gt_mask=gt_mask,
            unet_size=unet_size,
            native_size=native_size,
            margin_fraction=margin_fraction,
        )
        rows.append({
            "image_path": row["image_path"],
            "x1_native": float(box_native[0]),
            "y1_native": float(box_native[1]),
            "x2_native": float(box_native[2]),
            "y2_native": float(box_native[3]),
            "used_fallback": used_fallback,
            "source": row["source"],
            "split": row["split"],
        })

    oracle_df = pd.DataFrame(rows)
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    oracle_df.to_csv(output_csv_path, index=False)
    logger.info(f"Oracle box cache saved: {output_csv_path}")
    return oracle_df


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalized_to_pixel(
    stats: dict, native_size: int
) -> np.ndarray:
    """Convert normalized (0–1) box coords to pixel coords."""
    return np.array([
        stats["x1_norm"] * native_size,
        stats["y1_norm"] * native_size,
        stats["x2_norm"] * native_size,
        stats["y2_norm"] * native_size,
    ], dtype=np.float32)


def load_fallback_box_stats(json_path: str) -> dict:
    """Load fallback box statistics from JSON file."""
    with open(json_path) as f:
        stats = json.load(f)
    logger.info(
        f"Fallback box stats loaded from {json_path}: "
        f"({stats['x1_norm']:.3f},{stats['y1_norm']:.3f}) → "
        f"({stats['x2_norm']:.3f},{stats['y2_norm']:.3f})"
    )
    return stats
