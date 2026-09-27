"""
data/transforms.py

Image transformation pipelines for U-Net and SAM training/inference.

Key design decisions:
  - Augmentation applied to Shenzhen-train ONLY
  - Images use BILINEAR interpolation; masks use NEAREST (no interpolation artifacts)
  - Normalization stats loaded from pre-computed JSON (Shenzhen-train only)
  - SAM preprocessing handled by HuggingFace SamProcessor

Audit fix: MODERATE-17 — explicit interpolation modes per tensor type.
Audit fix: CRITICAL-06 — normalization uses only pre-computed stats.
"""

import json
import logging
from typing import Dict, Optional, Tuple

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)


def load_norm_stats(stats_json_path: str) -> Dict[str, list]:
    """
    Load pre-computed normalization statistics.

    Args:
        stats_json_path: Path to norm_stats.json from cleaner.py.

    Returns:
        Dict with 'mean' and 'std' as lists of 3 values.
    """
    with open(stats_json_path) as f:
        stats = json.load(f)
    logger.info(
        f"Loaded normalization stats: mean={stats['mean'][0]:.4f}, "
        f"std={stats['std'][0]:.4f}"
    )
    return stats


def get_unet_train_transforms(
    image_size: int,
    norm_stats: Optional[Dict[str, list]] = None,
    rotation_limit: int = 10,
    brightness_limit: float = 0.1,
    contrast_limit: float = 0.1,
    hflip_prob: float = 0.5,
    grid_distortion_prob: float = 0.2,
) -> A.Compose:
    """
    Augmentation + normalization pipeline for U-Net training.

    Applied to Shenzhen-train images only. Augmentations are applied
    identically to both image and mask using Albumentations' paired API.

    Args:
        image_size: Target H=W size for U-Net input.
        norm_stats: Pre-computed normalization stats dict.
                    If None, uses [0.5, 0.5, 0.5] as placeholder.
        rotation_limit: Max rotation in degrees.
        brightness_limit: Max brightness delta.
        contrast_limit: Max contrast delta.
        hflip_prob: Horizontal flip probability.
        grid_distortion_prob: Grid distortion probability.

    Returns:
        Albumentations Compose transform.
    """
    if norm_stats is None:
        mean = [0.5]
        std = [0.5]
        logger.warning("No normalization stats provided; using default [0.5, 0.5].")
    else:
        # Use single-channel stats for grayscale
        mean = [norm_stats["mean"][0]]
        std = [norm_stats["std"][0]]

    transforms = A.Compose([
        A.Resize(image_size, image_size,
                 interpolation=1,        # cv2.INTER_LINEAR for image
                 always_apply=True),
        A.HorizontalFlip(p=hflip_prob),
        A.Rotate(limit=rotation_limit, p=0.5,
                 interpolation=1,        # BILINEAR for image
                 border_mode=0),         # BORDER_CONSTANT
        A.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            p=0.5,
        ),
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.2,
            p=grid_distortion_prob,
            interpolation=1,
        ),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),
    ])
    return transforms


def get_unet_val_transforms(
    image_size: int,
    norm_stats: Optional[Dict[str, list]] = None,
) -> A.Compose:
    """
    Resize + normalization only for U-Net validation/inference.
    No augmentation. This is the only transform applied to val/test/Montgomery.

    Args:
        image_size: Target H=W size for U-Net input.
        norm_stats: Pre-computed normalization stats dict.

    Returns:
        Albumentations Compose transform.
    """
    if norm_stats is None:
        mean = [0.5]
        std = [0.5]
    else:
        mean = [norm_stats["mean"][0]]
        std = [norm_stats["std"][0]]

    transforms = A.Compose([
        A.Resize(image_size, image_size, interpolation=1, always_apply=True),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),
    ])
    return transforms


def get_mask_resize_transform(target_size: int) -> A.Compose:
    """
    Resize-only transform for masks using NEAREST interpolation.

    Used to resize GT masks to evaluation resolution.

    Args:
        target_size: Target H=W size.

    Returns:
        Albumentations Compose for mask-only transforms.
    """
    return A.Compose([
        A.Resize(target_size, target_size,
                 interpolation=0,        # cv2.INTER_NEAREST — no interpolation
                 always_apply=True),
    ])


def grayscale_to_rgb_numpy(image_gray: np.ndarray) -> np.ndarray:
    """
    Convert a single-channel grayscale image to 3-channel RGB.

    SAM's vision encoder (pre-trained on ImageNet) requires RGB input.
    We replicate the grayscale channel 3 times.

    Audit fix: CRITICAL-03 — explicit RGB conversion for SAM.

    Args:
        image_gray: numpy array of shape (H, W) or (H, W, 1), uint8 or float.

    Returns:
        numpy array of shape (H, W, 3), same dtype as input.
    """
    if image_gray.ndim == 3 and image_gray.shape[2] == 1:
        image_gray = image_gray[:, :, 0]
    if image_gray.ndim != 2:
        raise ValueError(
            f"Expected 2D grayscale image, got shape {image_gray.shape}"
        )
    return np.stack([image_gray, image_gray, image_gray], axis=-1)


def resize_image_pil(
    image_path: str,
    target_size: int,
    to_rgb: bool = False,
) -> np.ndarray:
    """
    Load and resize an image using PIL with BILINEAR interpolation.

    Args:
        image_path: Path to the image file.
        target_size: Target H=W size.
        to_rgb: If True, convert grayscale to 3-channel RGB.

    Returns:
        numpy uint8 array of shape (target_size, target_size) or
        (target_size, target_size, 3) if to_rgb=True.
    """
    from PIL import Image
    with Image.open(image_path) as img:
        img_gray = img.convert("L").resize(
            (target_size, target_size), Image.BILINEAR
        )
        arr = np.array(img_gray)

    if to_rgb:
        return grayscale_to_rgb_numpy(arr)
    return arr


def resize_mask_pil(mask_path: str, target_size: int) -> np.ndarray:
    """
    Load and resize a mask using PIL with NEAREST interpolation.

    Args:
        mask_path: Path to the mask file.
        target_size: Target H=W size.

    Returns:
        Binary numpy float32 array of shape (target_size, target_size).
    """
    from PIL import Image
    with Image.open(mask_path) as mimg:
        mask_resized = mimg.convert("L").resize(
            (target_size, target_size), Image.NEAREST
        )
        mask_arr = np.array(mask_resized, dtype=np.float32)

    # Normalize to [0, 1] binary
    if mask_arr.max() > 1:
        mask_arr = (mask_arr > 127).astype(np.float32)

    return mask_arr
