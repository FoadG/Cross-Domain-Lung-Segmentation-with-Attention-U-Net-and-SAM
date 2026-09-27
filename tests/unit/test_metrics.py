"""
tests/unit/test_metrics.py

Unit tests for evaluation/metrics.py.

Verifies:
  - Dice = 1.0 for perfect prediction, 0.0 for complement
  - IoU = 1.0 for perfect, 0.0 for complement
  - HD95 = 0.0 for identical masks, > 0 for offset masks
  - HD95 penalty for empty masks
  - Per-image computation matches manual calculation
  - aggregate_metrics computes correct mean and std
"""

import numpy as np
import pytest
import torch

from evaluation.metrics import (
    EMPTY_MASK_HD95_PENALTY,
    aggregate_metrics,
    compute_batch_dice,
    compute_batch_hd95,
    compute_batch_iou,
    compute_dice,
    compute_hd95,
    compute_iou,
)


class TestComputeDice:
    def test_perfect_prediction(self):
        mask = np.ones((256, 256), dtype=np.float32)
        assert compute_dice(mask, mask) == pytest.approx(1.0, abs=1e-5)

    def test_empty_prediction_nonempty_gt(self):
        pred = np.zeros((256, 256), dtype=np.float32)
        gt = np.ones((256, 256), dtype=np.float32)
        score = compute_dice(pred, gt)
        # With smooth=1e-6, result is near 0 but not exactly 0
        assert score < 0.01

    def test_complement_prediction(self):
        gt = np.zeros((128, 128), dtype=np.float32)
        gt[20:80, 20:80] = 1.0
        pred = 1.0 - gt
        score = compute_dice(pred, gt)
        # No overlap → Dice ≈ 0
        assert score < 0.01

    def test_partial_overlap(self):
        gt = np.zeros((100, 100), dtype=np.float32)
        gt[10:90, 10:90] = 1.0
        pred = np.zeros((100, 100), dtype=np.float32)
        pred[10:90, 50:90] = 1.0  # half of GT
        score = compute_dice(pred, gt)
        assert 0.5 < score < 0.8

    def test_both_empty(self):
        pred = np.zeros((64, 64), dtype=np.float32)
        gt = np.zeros((64, 64), dtype=np.float32)
        score = compute_dice(pred, gt)
        # Smooth prevents NaN; result should be near 1.0 (trivially "same")
        assert score >= 0.0


class TestComputeIoU:
    def test_perfect_prediction(self):
        mask = np.ones((256, 256), dtype=np.float32)
        assert compute_iou(mask, mask) == pytest.approx(1.0, abs=1e-5)

    def test_no_overlap(self):
        gt = np.zeros((100, 100), dtype=np.float32)
        gt[:50, :50] = 1.0
        pred = np.zeros((100, 100), dtype=np.float32)
        pred[50:, 50:] = 1.0
        score = compute_iou(pred, gt)
        assert score < 0.01

    def test_iou_leq_dice(self):
        """IoU ≤ Dice for any binary masks (mathematical property)."""
        gt = np.zeros((100, 100), dtype=np.float32)
        gt[10:80, 10:80] = 1.0
        pred = np.zeros((100, 100), dtype=np.float32)
        pred[20:90, 20:90] = 1.0
        dice = compute_dice(pred, gt)
        iou = compute_iou(pred, gt)
        assert iou <= dice + 1e-6


class TestComputeHD95:
    def test_identical_masks(self):
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[10:50, 10:50] = 1.0
        hd = compute_hd95(mask, mask)
        assert hd == pytest.approx(0.0, abs=1e-5)

    def test_empty_pred_returns_penalty(self):
        gt = np.zeros((64, 64), dtype=np.float32)
        gt[10:50, 10:50] = 1.0
        pred = np.zeros((64, 64), dtype=np.float32)
        hd = compute_hd95(pred, gt)
        # Resolution-aware penalty = image diagonal (audit fix #7), not a
        # hard-coded 512 tied to a single resolution.
        assert hd == pytest.approx(float(np.hypot(64, 64)), abs=1e-5)

    def test_empty_gt_returns_penalty(self):
        pred = np.zeros((64, 64), dtype=np.float32)
        pred[10:50, 10:50] = 1.0
        gt = np.zeros((64, 64), dtype=np.float32)
        hd = compute_hd95(pred, gt)
        assert hd == pytest.approx(float(np.hypot(64, 64)), abs=1e-5)

    def test_both_empty_returns_zero(self):
        empty = np.zeros((64, 64), dtype=np.float32)
        # Both masks empty => no boundary disagreement => 0.
        assert compute_hd95(empty, empty) == pytest.approx(0.0, abs=1e-5)

    def test_custom_empty_penalty_respected(self):
        gt = np.zeros((64, 64), dtype=np.float32)
        gt[10:50, 10:50] = 1.0
        pred = np.zeros((64, 64), dtype=np.float32)
        assert compute_hd95(pred, gt, empty_penalty=999.0) == pytest.approx(999.0)

    def test_offset_masks_positive_hd(self):
        pred = np.zeros((64, 64), dtype=np.float32)
        pred[10:30, 10:30] = 1.0
        gt = np.zeros((64, 64), dtype=np.float32)
        gt[30:50, 30:50] = 1.0  # shifted by ~28 pixels
        hd = compute_hd95(pred, gt)
        assert hd > 0

    def test_larger_offset_larger_hd(self):
        pred = np.zeros((128, 128), dtype=np.float32)
        pred[5:15, 5:15] = 1.0
        gt_near = np.zeros_like(pred)
        gt_near[10:20, 10:20] = 1.0
        gt_far = np.zeros_like(pred)
        gt_far[90:100, 90:100] = 1.0
        hd_near = compute_hd95(pred, gt_near)
        hd_far = compute_hd95(pred, gt_far)
        assert hd_far > hd_near


class TestBatchMetrics:
    def _make_batch(self, n=4, h=64, w=64, fill_fraction=0.5):
        preds = torch.zeros(n, 1, h, w)
        gts = torch.zeros(n, 1, h, w)
        hw = int(h * fill_fraction)
        ww = int(w * fill_fraction)
        preds[:, :, :hw, :ww] = 1.0
        gts[:, :, :hw, :ww] = 1.0
        return preds, gts

    def test_batch_dice_perfect(self):
        preds, gts = self._make_batch()
        scores = compute_batch_dice(preds, gts)
        assert len(scores) == 4
        assert all(s == pytest.approx(1.0, abs=1e-5) for s in scores)

    def test_batch_iou_perfect(self):
        preds, gts = self._make_batch()
        scores = compute_batch_iou(preds, gts)
        assert all(s == pytest.approx(1.0, abs=1e-5) for s in scores)

    def test_batch_hd95_returns_list(self):
        preds, gts = self._make_batch(n=2)
        scores = compute_batch_hd95(preds, gts)
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)

    def test_batch_dice_length_matches_batch_size(self):
        preds = torch.rand(7, 1, 32, 32)
        gts = (torch.rand(7, 1, 32, 32) > 0.5).float()
        scores = compute_batch_dice(preds > 0.5, gts)
        assert len(scores) == 7


class TestAggregateMetrics:
    def test_perfect_scores(self):
        dice = [1.0, 1.0, 1.0]
        iou = [1.0, 1.0, 1.0]
        result = aggregate_metrics(dice, iou)
        assert result["dice_mean"] == pytest.approx(1.0, abs=1e-6)
        assert result["dice_std"] == pytest.approx(0.0, abs=1e-6)
        assert result["n_images"] == 3

    def test_variable_scores(self):
        dice = [0.6, 0.8, 0.9]
        iou = [0.5, 0.7, 0.85]
        result = aggregate_metrics(dice, iou)
        assert result["dice_mean"] == pytest.approx(np.mean(dice), abs=1e-6)
        assert result["dice_std"] == pytest.approx(np.std(dice), abs=1e-6)

    def test_with_hd95(self):
        dice = [0.9, 0.85]
        iou = [0.82, 0.77]
        hd95 = [3.2, 5.1]
        result = aggregate_metrics(dice, iou, hd95)
        assert "hd95_mean" in result
        assert result["hd95_mean"] == pytest.approx(np.mean(hd95), abs=1e-5)

    def test_without_hd95(self):
        result = aggregate_metrics([0.9], [0.8])
        assert "hd95_mean" not in result
