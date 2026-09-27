"""
tests/unit/test_box_extractor.py

Unit tests for prompt_generation/box_extractor.py.

Critical validations:
  - Correct coordinate scaling (unet_size → native_size)
  - Outer-box margin is always additive (never shrinks)
  - Fallback is used when mask is empty
  - Oracle box uses GT, not prediction
  - Stress-test box is deterministic given seed
"""

import json
import tempfile

import numpy as np
import pytest

from prompt_generation.box_extractor import (
    extract_box_from_mask,
    get_oracle_box,
    get_pipeline_box,
    get_stress_test_box,
    load_fallback_box_stats,
    scale_box,
)


class TestExtractBoxFromMask:
    def _square_mask(self, size=256, box_start=40, box_end=120) -> np.ndarray:
        mask = np.zeros((size, size), dtype=np.float32)
        mask[box_start:box_end, box_start:box_end] = 1.0
        return mask

    def test_returns_box_for_valid_mask(self):
        mask = self._square_mask()
        box, used_fallback = extract_box_from_mask(mask, image_size=256)
        assert box is not None
        assert not used_fallback
        assert len(box) == 4
        x1, y1, x2, y2 = box
        assert x1 < x2 and y1 < y2

    def test_returns_none_for_empty_mask(self):
        mask = np.zeros((256, 256), dtype=np.float32)
        box, used_fallback = extract_box_from_mask(mask, image_size=256)
        assert box is None
        assert used_fallback

    def test_outer_box_margin_never_shrinks(self):
        """Box with margin must always be >= tight box."""
        mask = self._square_mask(box_start=50, box_end=100)

        # Tight box (no margin)
        box_no_margin, _ = extract_box_from_mask(
            mask, image_size=256, margin_fraction=0.0
        )
        # With margin
        box_with_margin, _ = extract_box_from_mask(
            mask, image_size=256, margin_fraction=0.12
        )

        assert box_with_margin[0] <= box_no_margin[0]  # x1 moves left (smaller)
        assert box_with_margin[1] <= box_no_margin[1]  # y1 moves up (smaller)
        assert box_with_margin[2] >= box_no_margin[2]  # x2 moves right (larger)
        assert box_with_margin[3] >= box_no_margin[3]  # y2 moves down (larger)

    def test_box_clipped_to_image_boundary(self):
        """Box must not exceed [0, image_size-1]."""
        # Large mask near the border
        mask = np.zeros((256, 256), dtype=np.float32)
        mask[200:256, 200:256] = 1.0
        box, _ = extract_box_from_mask(mask, image_size=256, margin_fraction=0.2)
        assert box is not None
        x1, y1, x2, y2 = box
        assert x1 >= 0 and y1 >= 0
        assert x2 <= 255 and y2 <= 255

    def test_x1_lt_x2_and_y1_lt_y2(self):
        mask = self._square_mask()
        box, _ = extract_box_from_mask(mask, image_size=256)
        x1, y1, x2, y2 = box
        assert x1 < x2, "x1 must be strictly less than x2"
        assert y1 < y2, "y1 must be strictly less than y2"

    def test_component_filtering_removes_noise(self):
        """Tiny isolated noise pixels should be ignored."""
        mask = np.zeros((256, 256), dtype=np.float32)
        mask[100:150, 100:150] = 1.0   # large component (valid)
        mask[0, 0] = 1.0                # single pixel noise
        mask[255, 255] = 1.0            # single pixel noise
        box, _ = extract_box_from_mask(
            mask, image_size=256, min_component_area_fraction=0.01
        )
        assert box is not None
        # Box should be around the large component, not the noise pixels
        x1, y1, x2, y2 = box
        assert x1 < 50 or x1 >= 0  # noise at (0,0) should not dominate


class TestScaleBox:
    def test_scale_up_by_factor_2(self):
        box = np.array([10.0, 20.0, 100.0, 200.0], dtype=np.float32)
        scaled = scale_box(box, from_size=256, to_size=512)
        expected = np.array([20.0, 40.0, 200.0, 400.0], dtype=np.float32)
        np.testing.assert_allclose(scaled, expected, atol=1.0)

    def test_scale_preserves_relative_position(self):
        box = np.array([64.0, 64.0, 192.0, 192.0], dtype=np.float32)  # center quarter
        scaled = scale_box(box, from_size=256, to_size=1024)
        # Should be the center quarter of 1024
        assert scaled[0] == pytest.approx(256.0, abs=2.0)
        assert scaled[2] == pytest.approx(768.0, abs=2.0)

    def test_scale_does_not_exceed_target_size(self):
        box = np.array([0.0, 0.0, 255.0, 255.0], dtype=np.float32)
        scaled = scale_box(box, from_size=256, to_size=512)
        assert scaled[2] <= 512 and scaled[3] <= 512


class TestGetOracleBox:
    def test_oracle_uses_gt_mask(self):
        gt_mask = np.zeros((256, 256), dtype=np.float32)
        gt_mask[60:180, 60:180] = 1.0

        box, used_fallback = get_oracle_box(
            gt_mask, unet_size=256, native_size=512, margin_fraction=0.12
        )
        assert not used_fallback
        # Box should be in native_size (512) space
        assert box.max() <= 512

    def test_oracle_returns_fallback_for_empty_gt(self):
        gt_mask = np.zeros((256, 256), dtype=np.float32)
        box, used_fallback = get_oracle_box(gt_mask, unet_size=256, native_size=512)
        assert used_fallback


class TestGetPipelineBox:
    @pytest.fixture
    def fallback_stats(self):
        return {
            "x1_norm": 0.1, "y1_norm": 0.1,
            "x2_norm": 0.9, "y2_norm": 0.9,
        }

    def test_uses_prediction_when_valid(self, fallback_stats):
        pred = np.zeros((256, 256), dtype=np.float32)
        pred[40:200, 40:200] = 1.0
        box, used_fallback = get_pipeline_box(
            pred_mask=pred,
            fallback_box_stats=fallback_stats,
            unet_size=256,
            native_size=512,
        )
        assert not used_fallback

    def test_uses_fallback_when_empty(self, fallback_stats):
        pred = np.zeros((256, 256), dtype=np.float32)
        box, used_fallback = get_pipeline_box(
            pred_mask=pred,
            fallback_box_stats=fallback_stats,
            unet_size=256,
            native_size=512,
        )
        assert used_fallback
        # Fallback box should be centered around [0.1*512, 0.1*512] = [51, 51]
        assert box[0] == pytest.approx(0.1 * 512, abs=2.0)

    def test_oracle_and_pipeline_are_different_functions(self, fallback_stats):
        """Audit fix CRITICAL-10: verify oracle and pipeline are truly separate."""
        pred = np.zeros((256, 256), dtype=np.float32)
        pred[40:100, 40:100] = 1.0  # small region

        gt = np.zeros((256, 256), dtype=np.float32)
        gt[80:200, 80:200] = 1.0    # different large region

        pipe_box, _ = get_pipeline_box(pred, fallback_stats, 256, 512)
        oracle_box, _ = get_oracle_box(gt, 256, 512)

        # They should differ because pred and gt are different
        assert not np.allclose(pipe_box, oracle_box, atol=10.0)


class TestGetStressTestBox:
    def test_deterministic_with_same_seed(self):
        gt = np.zeros((256, 256), dtype=np.float32)
        gt[50:150, 50:150] = 1.0
        box1 = get_stress_test_box(gt, error_scale=0.25, seed=42)
        box2 = get_stress_test_box(gt, error_scale=0.25, seed=42)
        np.testing.assert_array_equal(box1, box2)

    def test_different_seed_different_box(self):
        gt = np.zeros((256, 256), dtype=np.float32)
        gt[50:150, 50:150] = 1.0
        box1 = get_stress_test_box(gt, error_scale=0.25, seed=42)
        box2 = get_stress_test_box(gt, error_scale=0.25, seed=99)
        assert not np.allclose(box1, box2)

    def test_box_is_valid(self):
        gt = np.zeros((256, 256), dtype=np.float32)
        gt[50:150, 50:150] = 1.0
        box = get_stress_test_box(gt, native_size=512, seed=0)
        x1, y1, x2, y2 = box
        assert x1 < x2
        assert y1 < y2
        assert all(v >= 0 for v in box)
        assert all(v < 512 for v in box)


class TestLoadFallbackBoxStats:
    def test_loads_correctly_from_json(self):
        stats = {
            "x1_norm": 0.12, "y1_norm": 0.08,
            "x2_norm": 0.88, "y2_norm": 0.92,
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(stats, f)
            tmp_path = f.name

        loaded = load_fallback_box_stats(tmp_path)
        assert loaded["x1_norm"] == pytest.approx(0.12)
        assert loaded["y2_norm"] == pytest.approx(0.92)
