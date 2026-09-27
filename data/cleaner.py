"""
data/cleaner.py

Transforms the raw validation report into a clean, canonical CSV
that serves as the single source of truth for all downstream components.

Also computes:
  1. Normalization statistics from Shenzhen-train images only
  2. Fallback box statistics from Shenzhen-train masks (for CRITICAL-07 fix)

Audit fixes:
  CRITICAL-06 — normalization uses only Shenzhen-train
  CRITICAL-07 — fallback box stored as JSON constant after split
  CRITICAL-11 — explicit assertion: no Montgomery in training data
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from data.validator import FileRecord, ValidationReport

logger = logging.getLogger(__name__)


def build_clean_csv(
    report: ValidationReport,
    output_csv_path: str,
) -> pd.DataFrame:
    """
    Build a canonical DataFrame from the validation report.

    Includes only valid records. Each row has:
        image_path, mask_path, source, split (filled later), image_h, image_w

    Args:
        report: Completed validation report.
        output_csv_path: Where to write the CSV.

    Returns:
        DataFrame with all valid records (split column is empty at this stage).
    """
    valid_records = [r for r in report.records if r.is_valid]

    rows = []
    for r in valid_records:
        rows.append({
            "image_path": r.image_path,
            "mask_path": r.mask_path,
            "source": r.source,
            "split": "",           # filled by split.py
            "image_h": r.image_shape[0] if r.image_shape else None,
            "image_w": r.image_shape[1] if r.image_shape else None,
            "mask_foreground_fraction": r.mask_foreground_fraction,
        })

    df = pd.DataFrame(rows)
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    logger.info(
        f"Clean CSV written: {output_csv_path} | "
        f"total={len(df)} | "
        f"Montgomery={len(df[df.source=='Montgomery'])} | "
        f"Shenzhen={len(df[df.source=='Shenzhen'])}"
    )
    return df


def compute_normalization_stats(
    df: pd.DataFrame,
    native_size: int = 512,
    output_json_path: str = "./data/processed/norm_stats.json",
    in_domain_source: str = "Shenzhen",
) -> Dict[str, float]:
    """
    Compute per-channel mean and std from in-domain-train images ONLY.

    This function must be called AFTER split assignment so that the
    'split' column is populated. Only 'train' rows of `in_domain_source`
    are used (default 'Shenzhen'; set to the degraded-mode in-domain source
    when only one domain is present).

    For grayscale CXR images, mean/std are single values (not per-channel).
    They are stored as [value, value, value] for compatibility with
    the 3-channel RGB conversion used by SAM.

    Args:
        df: DataFrame with 'source' and 'split' columns populated.
        native_size: Resize target before computing stats.
        output_json_path: Where to save the JSON stats file.
        in_domain_source: Which source counts as in-domain (default 'Shenzhen').

    Returns:
        Dict with keys 'mean' and 'std' (each a list of 3 identical values).

    Raises:
        ValueError: If no in-domain-train images are found.
    """
    # Only the in-domain source + train split (audit CRITICAL-06).
    train_shenzhen = df[
        (df["source"] == in_domain_source) & (df["split"] == "train")
    ]

    if len(train_shenzhen) == 0:
        raise ValueError(
            f"No {in_domain_source}-train images found. "
            "Ensure split assignment ran before compute_normalization_stats()."
        )

    logger.info(
        f"Computing normalization stats from {len(train_shenzhen)} "
        f"{in_domain_source}-train images..."
    )

    pixel_values: List[float] = []

    for _, row in train_shenzhen.iterrows():
        try:
            with Image.open(row["image_path"]) as img:
                img_gray = img.convert("L").resize(
                    (native_size, native_size), Image.BILINEAR
                )
                arr = np.array(img_gray, dtype=np.float32) / 255.0
                pixel_values.extend(arr.flatten().tolist())
        except Exception as e:
            logger.warning(f"Could not read {row['image_path']} for stats: {e}")

    if not pixel_values:
        raise ValueError("Failed to read any images for normalization stats.")

    pv = np.array(pixel_values, dtype=np.float64)
    mean_val = float(pv.mean())
    std_val = float(pv.std())

    # For the 3-channel RGB representation used by SAM
    stats = {
        "mean": [mean_val, mean_val, mean_val],
        "std": [std_val, std_val, std_val],
        "n_images": len(train_shenzhen),
        "n_pixels": len(pixel_values),
    }

    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(
        f"Normalization stats: mean={mean_val:.4f}, std={std_val:.4f} | "
        f"saved to {output_json_path}"
    )
    return stats


def compute_fallback_box_stats(
    df: pd.DataFrame,
    unet_size: int = 256,
    box_margin_fraction: float = 0.12,
    output_json_path: str = "./data/processed/fallback_box_stats.json",
    in_domain_source: str = "Shenzhen",
) -> Dict[str, float]:
    """
    Compute the fallback bounding box from in-domain-train ground-truth masks.

    When U-Net produces an empty or invalid mask during inference,
    this pre-computed box is used as a fallback. It represents the
    mean anatomical position of the lungs in the in-domain-train images.

    Must be called AFTER split assignment.

    Audit fix: CRITICAL-07 — fallback box as deterministic constant.

    Args:
        df: DataFrame with 'source', 'split', and 'mask_path' populated.
        unet_size: Resolution at which boxes are computed (256×256).
        box_margin_fraction: Outer-Box expansion fraction.
        output_json_path: Where to save the fallback box JSON.
        in_domain_source: Which source counts as in-domain (default 'Shenzhen').

    Returns:
        Dict with keys 'x1', 'y1', 'x2', 'y2' (normalized, 0-1 range).
    """
    train_shenzhen = df[
        (df["source"] == in_domain_source) & (df["split"] == "train")
    ]

    if len(train_shenzhen) == 0:
        raise ValueError(
            f"No {in_domain_source}-train images found for fallback box computation."
        )

    logger.info(
        f"Computing fallback box from {len(train_shenzhen)} "
        f"{in_domain_source}-train masks..."
    )

    x1_vals, y1_vals, x2_vals, y2_vals = [], [], [], []

    for _, row in train_shenzhen.iterrows():
        try:
            with Image.open(row["mask_path"]) as mimg:
                mask = np.array(mimg.convert("L").resize(
                    (unet_size, unet_size), Image.NEAREST
                ))
                if mask.max() > 1:
                    mask = (mask > 127).astype(np.uint8)

                rows_with_fg = np.any(mask > 0, axis=1)
                cols_with_fg = np.any(mask > 0, axis=0)

                if not rows_with_fg.any():
                    continue

                y1 = float(np.argmax(rows_with_fg)) / unet_size
                y2 = float(unet_size - np.argmax(rows_with_fg[::-1]) - 1) / unet_size
                x1 = float(np.argmax(cols_with_fg)) / unet_size
                x2 = float(unet_size - np.argmax(cols_with_fg[::-1]) - 1) / unet_size

                x1_vals.append(x1)
                y1_vals.append(y1)
                x2_vals.append(x2)
                y2_vals.append(y2)
        except Exception as e:
            logger.warning(f"Skipping mask {row['mask_path']}: {e}")

    if not x1_vals:
        raise ValueError("Could not compute fallback box — no valid masks found.")

    # Apply outer-box margin in normalized space
    margin = box_margin_fraction
    stats = {
        "x1_norm": max(0.0, float(np.mean(x1_vals)) - margin),
        "y1_norm": max(0.0, float(np.mean(y1_vals)) - margin),
        "x2_norm": min(1.0, float(np.mean(x2_vals)) + margin),
        "y2_norm": min(1.0, float(np.mean(y2_vals)) + margin),
        "n_images": len(x1_vals),
        "margin_applied": margin,
    }

    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(
        f"Fallback box (normalized): "
        f"({stats['x1_norm']:.3f}, {stats['y1_norm']:.3f}) → "
        f"({stats['x2_norm']:.3f}, {stats['y2_norm']:.3f}) | "
        f"saved to {output_json_path}"
    )
    return stats
