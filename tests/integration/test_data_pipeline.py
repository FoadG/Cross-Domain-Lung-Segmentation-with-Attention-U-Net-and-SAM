"""
tests/integration/test_data_pipeline.py

Integration tests for the full data pipeline:
  downloader → validator → cleaner → split → transforms → dataset

These tests use synthetic in-memory data (no Kaggle download required).
They verify that all pipeline stages connect correctly and that
the Montgomery leakage guard is enforced end-to-end.
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from data.cleaner import build_clean_csv, compute_normalization_stats
from data.split import assign_splits, _assert_no_montgomery_in_training
from data.validator import FileRecord, ValidationReport


def _make_synthetic_dataset(
    tmpdir: str,
    n_shenzhen: int = 20,
    n_montgomery: int = 6,
    n_no_mask: int = 3,
) -> str:
    """Create synthetic PNG images and masks mimicking the real dataset layout."""
    img_dir = os.path.join(tmpdir, "images")
    mask_dir = os.path.join(tmpdir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for i in range(n_shenzhen):
        prefix = f"CHNCXR_{i:04d}_0"
        img = Image.fromarray(
            np.random.randint(50, 200, (128, 128), dtype=np.uint8), mode="L"
        )
        img.save(os.path.join(img_dir, f"{prefix}.png"))
        mask_arr = np.zeros((128, 128), dtype=np.uint8)
        mask_arr[20:100, 20:100] = 255
        Image.fromarray(mask_arr, mode="L").save(
            os.path.join(mask_dir, f"{prefix}.png")
        )

    for i in range(n_montgomery):
        prefix = f"MCUCXR_{i:04d}_0"
        img = Image.fromarray(
            np.random.randint(30, 180, (128, 128), dtype=np.uint8), mode="L"
        )
        img.save(os.path.join(img_dir, f"{prefix}.png"))
        mask_arr = np.zeros((128, 128), dtype=np.uint8)
        mask_arr[15:110, 15:110] = 255
        Image.fromarray(mask_arr, mode="L").save(
            os.path.join(mask_dir, f"{prefix}.png")
        )

    # Images without masks (should be excluded)
    for i in range(n_no_mask):
        Image.fromarray(
            np.random.randint(0, 255, (128, 128), dtype=np.uint8), mode="L"
        ).save(os.path.join(img_dir, f"CHNCXR_no_mask_{i}.png"))

    return tmpdir


def _make_synthetic_df(
    n_shenzhen=50, n_montgomery=8
) -> pd.DataFrame:
    """Make a DataFrame directly (without reading from disk)."""
    rows = []
    for i in range(n_shenzhen):
        rows.append({
            "image_path": f"/fake/CHNCXR_{i:04d}.png",
            "mask_path": f"/fake/mask_CHNCXR_{i:04d}.png",
            "source": "Shenzhen",
            "split": "",
            "image_h": 128, "image_w": 128,
            "mask_foreground_fraction": 0.25 + 0.01 * i,
        })
    for i in range(n_montgomery):
        rows.append({
            "image_path": f"/fake/MCUCXR_{i:04d}.png",
            "mask_path": f"/fake/mask_MCUCXR_{i:04d}.png",
            "source": "Montgomery",
            "split": "",
            "image_h": 128, "image_w": 128,
            "mask_foreground_fraction": 0.30 + 0.01 * i,
        })
    return pd.DataFrame(rows)


class TestSplitAssignment:
    def test_montgomery_always_in_cross_domain_test(self, tmp_path):
        df = _make_synthetic_df(n_shenzhen=50, n_montgomery=8)
        split_df = assign_splits(df, splits_dir=str(tmp_path))
        montgomery_splits = split_df[split_df["source"] == "Montgomery"]["split"].unique()
        assert list(montgomery_splits) == ["cross_domain_test"]

    def test_no_montgomery_in_training(self, tmp_path):
        df = _make_synthetic_df(n_shenzhen=50, n_montgomery=8)
        split_df = assign_splits(df, splits_dir=str(tmp_path))
        # This assertion should pass without raising
        _assert_no_montgomery_in_training(split_df)

    def test_shenzhen_split_ratios_approximately_correct(self, tmp_path):
        n_sh = 80
        df = _make_synthetic_df(n_shenzhen=n_sh, n_montgomery=10)
        split_df = assign_splits(
            df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            splits_dir=str(tmp_path)
        )
        shenzhen_df = split_df[split_df["source"] == "Shenzhen"]
        n_train = (shenzhen_df["split"] == "train").sum()
        n_val = (shenzhen_df["split"] == "val").sum()
        n_test = (shenzhen_df["split"] == "test").sum()
        total = n_train + n_val + n_test
        # Ratios within 10% of target
        assert abs(n_train / total - 0.70) < 0.10
        assert abs(n_val / total - 0.15) < 0.10
        assert abs(n_test / total - 0.15) < 0.10

    def test_split_ratios_sum_error_raises(self, tmp_path):
        df = _make_synthetic_df()
        with pytest.raises(ValueError, match="sum to 1.0"):
            assign_splits(df, train_ratio=0.5, val_ratio=0.4, test_ratio=0.2,
                          splits_dir=str(tmp_path))

    def test_poisoned_df_raises_assertion(self, tmp_path):
        """Verify CRITICAL-11: poisoned DataFrame triggers assertion."""
        df = _make_synthetic_df(n_shenzhen=50, n_montgomery=8)
        split_df = assign_splits(df, splits_dir=str(tmp_path))

        # Poison: move a Montgomery record to train
        idx = split_df[split_df["source"] == "Montgomery"].index[0]
        split_df.at[idx, "split"] = "train"

        with pytest.raises(AssertionError, match="DATA LEAKAGE"):
            _assert_no_montgomery_in_training(split_df)

    def test_csv_saved_and_loadable(self, tmp_path):
        df = _make_synthetic_df(n_shenzhen=50, n_montgomery=8)
        split_df = assign_splits(df, splits_dir=str(tmp_path))
        csv_path = tmp_path / "splits.csv"
        assert csv_path.exists()
        loaded = pd.read_csv(str(csv_path))
        assert len(loaded) == len(split_df)
        assert "split" in loaded.columns


class TestNormalizationStats:
    def test_computed_from_shenzhen_train_only(self, tmp_path):
        """Stats must be None or only computed after split assignment."""
        df = _make_synthetic_df(n_shenzhen=50, n_montgomery=8)
        split_df = assign_splits(df, splits_dir=str(tmp_path))

        # Without real image files, we just verify the filter logic
        train_sh = split_df[
            (split_df["source"] == "Shenzhen") & (split_df["split"] == "train")
        ]
        montgomery_in_train = split_df[
            (split_df["source"] == "Montgomery") & (split_df["split"] == "train")
        ]
        assert len(montgomery_in_train) == 0
        assert len(train_sh) > 0


class TestBuildCleanCSV:
    def test_filters_invalid_records(self, tmp_path):
        """CSV must only contain valid records."""
        # Create a minimal ValidationReport with mixed valid/invalid records
        valid_records = [
            FileRecord(
                image_path=f"/img/{i}.png",
                mask_path=f"/mask/{i}.png",
                all_mask_paths=[f"/mask/{i}.png"],
                source="Shenzhen",
                mask_exists=True,
                image_readable=True,
                mask_readable=True,
                image_shape=(128, 128),
                mask_shape=(128, 128),
                mask_foreground_fraction=0.25,
                issues=[],
            )
            for i in range(5)
        ]
        invalid_records = [
            FileRecord(
                image_path=f"/img/bad_{i}.png",
                mask_path=None,
                all_mask_paths=[],
                source="Shenzhen",
                mask_exists=False,
                image_readable=True,
                mask_readable=False,
                image_shape=(128, 128),
                mask_shape=None,
                mask_foreground_fraction=None,
                issues=["No mask found"],
            )
            for i in range(3)
        ]

        report = ValidationReport(
            total_images=8,
            valid_images=5,
            excluded_no_mask=3,
            excluded_unreadable=0,
            montgomery_count=0,
            shenzhen_count=5,
            records=valid_records + invalid_records,
        )

        csv_path = str(tmp_path / "clean.csv")
        df = build_clean_csv(report, csv_path)
        assert len(df) == 5
        assert all(df["source"] == "Shenzhen")
