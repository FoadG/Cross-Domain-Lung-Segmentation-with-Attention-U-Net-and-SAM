"""
data/dataset.py

PyTorch Dataset implementations for U-Net training and SAM fine-tuning.

Design:
  - UNetDataset: single-channel grayscale images, binary masks
  - SAMDataset: RGB images (grayscale replicated 3×), binary masks, box prompts
  - Both include runtime Montgomery-in-training assertion (audit fix CRITICAL-11)

Audit fix: CRITICAL-11 — Montgomery assertion in __getitem__.
Audit fix: CRITICAL-03 — grayscale→RGB for SAM.
Audit fix: CRITICAL-14 — caching logic for SAM image embeddings.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.transforms import (
    grayscale_to_rgb_numpy,
    resize_image_pil,
    resize_mask_pil,
)

logger = logging.getLogger(__name__)


class UNetDataset(Dataset):
    """
    Dataset for U-Net training and inference.

    Loads grayscale CXR images and binary lung masks.
    Applies per-split transform pipeline.

    Args:
        df: DataFrame subset for this split (already filtered to correct split).
        transform: Albumentations Compose transform for image+mask.
        split_name: Human-readable split name for error messages.
        assert_no_montgomery_in_training: If True, raises AssertionError
            if any Montgomery image is loaded during a training split.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform,
        split_name: str = "train",
        assert_no_montgomery_in_training: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.split_name = split_name
        self._is_training_split = split_name in ("train", "val")
        self._assert_leakage = (
            assert_no_montgomery_in_training and self._is_training_split
        )

        logger.info(
            f"UNetDataset: split='{split_name}', "
            f"n={len(self.df)}, "
            f"sources={self.df['source'].unique().tolist()}"
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # CRITICAL-11: runtime leakage guard
        if self._assert_leakage and row["source"] == "Montgomery":
            raise AssertionError(
                f"DATA LEAKAGE: Montgomery image found in '{self.split_name}' "
                f"split at index {idx}: {row['image_path']}"
            )

        # Load grayscale image
        image_pil_arr = _load_image_gray(row["image_path"])
        mask_arr = _load_mask(row["mask_path"])

        # Apply Albumentations transform (handles both image and mask)
        transformed = self.transform(
            image=image_pil_arr,
            mask=mask_arr,
        )
        image_tensor = transformed["image"]   # shape: (1, H, W), float32
        mask_tensor = transformed["mask"]     # shape: (H, W), float32

        # Add channel dim to mask for loss computation
        mask_tensor = mask_tensor.unsqueeze(0)  # (1, H, W)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "source": row["source"],
            "image_path": row["image_path"],
            "split": row["split"],
        }


class SAMDataset(Dataset):
    """
    Dataset for SAM fine-tuning.

    Returns:
        - rgb_image: np.ndarray (H, W, 3) uint8 — for SamProcessor
        - gt_mask: torch.Tensor (1, H, W) float32 — ground truth
        - box: np.ndarray [x1, y1, x2, y2] in native_size pixel space
        - source: "Shenzhen" or "Montgomery"
        - image_path: file path string

    The SamProcessor converts rgb_image and box to model inputs.
    Collation is manual (see sam_collate_fn) since images may vary in
    processor output shapes before batching.

    Args:
        df: DataFrame subset for this split.
        box_cache_df: DataFrame with pre-computed boxes (from box_extractor.py).
        native_size: Resolution for rgb_image and box coordinates.
        split_name: Human-readable split name.
        assert_no_montgomery_in_training: Leakage guard.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        box_cache_df: pd.DataFrame,
        native_size: int = 512,
        split_name: str = "train",
        assert_no_montgomery_in_training: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.native_size = native_size
        self.split_name = split_name
        self._is_training_split = split_name in ("train", "val")
        self._assert_leakage = (
            assert_no_montgomery_in_training and self._is_training_split
        )

        # Build lookup: image_path → box row
        self._box_lookup = {
            row["image_path"]: row
            for _, row in box_cache_df.iterrows()
        }

        logger.info(
            f"SAMDataset: split='{split_name}', n={len(self.df)}, "
            f"sources={self.df['source'].unique().tolist()}"
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        image_path = row["image_path"]

        # CRITICAL-11: runtime leakage guard
        if self._assert_leakage and row["source"] == "Montgomery":
            raise AssertionError(
                f"DATA LEAKAGE: Montgomery image in '{self.split_name}' "
                f"at index {idx}: {image_path}"
            )

        # Load RGB image for SAM (grayscale → 3-channel)
        rgb_image = resize_image_pil(
            image_path, self.native_size, to_rgb=True
        )  # (native_size, native_size, 3), uint8

        # Load ground-truth mask
        gt_mask = resize_mask_pil(row["mask_path"], self.native_size)
        gt_mask_tensor = torch.from_numpy(gt_mask).unsqueeze(0)  # (1, H, W)

        # Get box from cache
        box = self._get_box(image_path)  # [x1, y1, x2, y2] in native_size space

        return {
            "rgb_image": rgb_image,        # np.ndarray (H, W, 3) uint8
            "gt_mask": gt_mask_tensor,     # Tensor (1, H, W) float32
            "box": box,                    # np.ndarray [x1, y1, x2, y2]
            "source": row["source"],
            "image_path": image_path,
            "split": row["split"],
        }

    def _get_box(self, image_path: str) -> np.ndarray:
        """
        Retrieve pre-computed bounding box for an image.

        Returns box in native_size pixel coordinates [x1, y1, x2, y2].
        If no box found, returns a broad default box (full image, minus margin).
        """
        if image_path in self._box_lookup:
            box_row = self._box_lookup[image_path]
            return np.array([
                float(box_row["x1_native"]),
                float(box_row["y1_native"]),
                float(box_row["x2_native"]),
                float(box_row["y2_native"]),
            ], dtype=np.float32)
        else:
            logger.warning(
                f"No cached box for {image_path}. Using full-image fallback."
            )
            margin = int(0.12 * self.native_size)
            return np.array([
                margin, margin,
                self.native_size - margin,
                self.native_size - margin,
            ], dtype=np.float32)


def sam_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for SAMDataset.

    rgb_image and box are kept as lists (not stacked) because
    SamProcessor handles batching internally.

    Args:
        batch: List of dicts from SAMDataset.__getitem__.

    Returns:
        Collated dict with:
            - rgb_images: List[np.ndarray] (one per item)
            - gt_masks: Tensor (B, 1, H, W)
            - boxes: List[np.ndarray] (one per item)
            - sources: List[str]
            - image_paths: List[str]
    """
    return {
        "rgb_images": [item["rgb_image"] for item in batch],
        "gt_masks": torch.stack([item["gt_mask"] for item in batch]),
        "boxes": [item["box"] for item in batch],
        "sources": [item["source"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
    }


def _load_image_gray(image_path: str) -> np.ndarray:
    """Load grayscale image as numpy uint8 array."""
    from PIL import Image
    with Image.open(image_path) as img:
        return np.array(img.convert("L"), dtype=np.uint8)


def _load_mask(mask_path: str) -> np.ndarray:
    """Load binary mask as numpy float32 array."""
    from PIL import Image
    with Image.open(mask_path) as mimg:
        arr = np.array(mimg.convert("L"), dtype=np.float32)
    if arr.max() > 1:
        arr = (arr > 127).astype(np.float32)
    return arr


def _load_combined_montgomery_mask(
    mask_paths: List[str], target_size: Optional[int] = None
) -> np.ndarray:
    """
    Load and combine Montgomery left+right lung masks.

    Args:
        mask_paths: Paths to left and right lung masks.
        target_size: Optional resize target.

    Returns:
        Combined binary float32 mask.
    """
    from PIL import Image
    combined = None
    for mp in mask_paths:
        with Image.open(mp) as mimg:
            if target_size:
                mimg = mimg.convert("L").resize(
                    (target_size, target_size), Image.NEAREST
                )
            arr = np.array(mimg.convert("L"), dtype=np.float32)
            if arr.max() > 1:
                arr = (arr > 127).astype(np.float32)
            combined = arr if combined is None else np.maximum(combined, arr)
    return combined.astype(np.float32)
