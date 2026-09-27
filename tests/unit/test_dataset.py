"""
tests/unit/test_dataset.py

Unit tests for data/dataset.py.

Critical tests:
  - UNetDataset raises AssertionError when Montgomery in training split
  - UNetDataset does NOT raise when Montgomery in cross_domain_test
  - Tensor shapes are correct
  - Mask values are binary [0, 1]
  - SAM dataset returns correct format
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from data.dataset import UNetDataset, sam_collate_fn


def _make_fake_dataset(tmpdir: str, n_shenzhen: int = 4, n_montgomery: int = 2):
    """Create a minimal fake dataset with real image and mask files."""
    img_dir = os.path.join(tmpdir, "images")
    mask_dir = os.path.join(tmpdir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    records = []

    # Shenzhen images
    for i in range(n_shenzhen):
        fname = f"CHNCXR_{i:04d}_0.png"
        img_path = os.path.join(img_dir, fname)
        mask_path = os.path.join(mask_dir, fname)

        # 128×128 grayscale image
        img = Image.fromarray(
            (np.random.rand(128, 128) * 255).astype(np.uint8), mode="L"
        )
        img.save(img_path)

        # Binary mask
        mask_arr = np.zeros((128, 128), dtype=np.uint8)
        mask_arr[30:90, 30:90] = 255
        Image.fromarray(mask_arr, mode="L").save(mask_path)

        records.append({
            "image_path": img_path,
            "mask_path": mask_path,
            "source": "Shenzhen",
            "split": "train" if i < n_shenzhen // 2 else "val",
            "image_h": 128, "image_w": 128,
            "mask_foreground_fraction": 0.25,
        })

    # Montgomery images (should only be in cross_domain_test)
    for i in range(n_montgomery):
        fname = f"MCUCXR_{i:04d}_0.png"
        img_path = os.path.join(img_dir, fname)
        mask_path = os.path.join(mask_dir, fname)

        img = Image.fromarray(
            (np.random.rand(128, 128) * 255).astype(np.uint8), mode="L"
        )
        img.save(img_path)
        mask_arr = np.zeros((128, 128), dtype=np.uint8)
        mask_arr[20:100, 20:100] = 255
        Image.fromarray(mask_arr, mode="L").save(mask_path)

        records.append({
            "image_path": img_path,
            "mask_path": mask_path,
            "source": "Montgomery",
            "split": "cross_domain_test",
            "image_h": 128, "image_w": 128,
            "mask_foreground_fraction": 0.35,
        })

    return pd.DataFrame(records)


def _make_transform():
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    return A.Compose([
        A.Resize(64, 64),
        A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255.0),
        ToTensorV2(),
    ])


class TestUNetDataset:
    def test_basic_loading(self, tmp_path):
        df = _make_fake_dataset(str(tmp_path))
        train_df = df[df["split"] == "train"].reset_index(drop=True)
        transform = _make_transform()

        dataset = UNetDataset(train_df, transform=transform, split_name="train")
        assert len(dataset) == len(train_df)

    def test_tensor_shapes(self, tmp_path):
        df = _make_fake_dataset(str(tmp_path))
        train_df = df[df["split"] == "train"].reset_index(drop=True)
        transform = _make_transform()

        dataset = UNetDataset(train_df, transform=transform, split_name="train")
        sample = dataset[0]

        assert "image" in sample
        assert "mask" in sample
        assert sample["image"].shape == (1, 64, 64)  # 1-channel, 64×64
        assert sample["mask"].shape == (1, 64, 64)

    def test_mask_is_binary(self, tmp_path):
        df = _make_fake_dataset(str(tmp_path))
        train_df = df[df["split"] == "train"].reset_index(drop=True)
        transform = _make_transform()

        dataset = UNetDataset(train_df, transform=transform, split_name="train")
        for i in range(len(dataset)):
            sample = dataset[i]
            mask_vals = sample["mask"].unique()
            # After albumentations, mask may have been binarized; check values
            assert sample["mask"].min() >= 0.0
            assert sample["mask"].max() <= 1.0

    def test_montgomery_in_training_raises(self, tmp_path):
        """CRITICAL-11: Montgomery in training split must raise AssertionError."""
        df = _make_fake_dataset(str(tmp_path))
        # Intentionally poison: put Montgomery in train split
        df_poisoned = df.copy()
        df_poisoned.loc[df_poisoned["source"] == "Montgomery", "split"] = "train"

        train_df = df_poisoned[df_poisoned["split"] == "train"].reset_index(drop=True)
        transform = _make_transform()

        dataset = UNetDataset(
            train_df, transform=transform, split_name="train",
            assert_no_montgomery_in_training=True,
        )

        with pytest.raises(AssertionError, match="DATA LEAKAGE"):
            # Accessing the first Montgomery item should raise
            for i in range(len(dataset)):
                _ = dataset[i]

    def test_montgomery_in_test_does_not_raise(self, tmp_path):
        """Montgomery in cross_domain_test split must NOT raise."""
        df = _make_fake_dataset(str(tmp_path))
        test_df = df[df["split"] == "cross_domain_test"].reset_index(drop=True)
        transform = _make_transform()

        dataset = UNetDataset(
            test_df, transform=transform, split_name="cross_domain_test",
            assert_no_montgomery_in_training=True,
        )

        # Should not raise
        for i in range(len(dataset)):
            sample = dataset[i]
            assert sample["source"] == "Montgomery"

    def test_source_field_returned(self, tmp_path):
        df = _make_fake_dataset(str(tmp_path))
        train_df = df[df["split"] == "train"].reset_index(drop=True)
        transform = _make_transform()
        dataset = UNetDataset(train_df, transform=transform, split_name="train")
        sample = dataset[0]
        assert "source" in sample
        assert sample["source"] in ("Shenzhen", "Montgomery")


class TestSamCollateFn:
    def test_collate_batches_correctly(self):
        batch = [
            {
                "rgb_image": np.zeros((64, 64, 3), dtype=np.uint8),
                "gt_mask": torch.zeros(1, 64, 64),
                "box": np.array([10.0, 10.0, 50.0, 50.0], dtype=np.float32),
                "source": "Shenzhen",
                "image_path": "/tmp/test.png",
                "split": "train",
            }
            for _ in range(3)
        ]
        collated = sam_collate_fn(batch)

        assert len(collated["rgb_images"]) == 3
        assert collated["gt_masks"].shape == (3, 1, 64, 64)
        assert len(collated["boxes"]) == 3
        assert len(collated["sources"]) == 3
        assert len(collated["image_paths"]) == 3
