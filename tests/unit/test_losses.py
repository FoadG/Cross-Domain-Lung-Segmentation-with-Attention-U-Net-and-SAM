"""
tests/unit/test_losses.py

Unit tests for training/losses.py.

Validates:
  - DiceLoss = 0 for perfect prediction
  - DiceLoss = 1 (max) for complement prediction
  - Per-image computation (not global pixels)
  - CombinedLoss returns components dict
  - Alpha weighting is correct
  - Factory function creates correct type
"""

import pytest
import torch

from training.losses import BCEWithLogitsLoss, CombinedLoss, DiceLoss, build_loss


class TestDiceLoss:
    def _perfect_batch(self, B=2, H=64, W=64):
        """Perfect prediction: logits → sigmoid → 1.0, gt = 1.0."""
        # Large positive logit → sigmoid ≈ 1.0
        logits = torch.ones(B, 1, H, W) * 10.0
        targets = torch.ones(B, 1, H, W)
        return logits, targets

    def _worst_batch(self, B=2, H=64, W=64):
        """Inverted prediction: logits → sigmoid ≈ 0, gt = 1.0."""
        logits = torch.ones(B, 1, H, W) * -10.0
        targets = torch.ones(B, 1, H, W)
        return logits, targets

    def test_zero_for_perfect_prediction(self):
        criterion = DiceLoss()
        logits, targets = self._perfect_batch()
        loss = criterion(logits, targets)
        assert loss.item() == pytest.approx(0.0, abs=1e-3)

    def test_max_for_complement_prediction(self):
        criterion = DiceLoss()
        logits, targets = self._worst_batch()
        loss = criterion(logits, targets)
        assert loss.item() > 0.95

    def test_between_zero_and_one(self):
        criterion = DiceLoss()
        logits = torch.rand(4, 1, 64, 64)
        targets = (torch.rand(4, 1, 64, 64) > 0.5).float()
        loss = criterion(logits, targets)
        assert 0.0 <= loss.item() <= 1.0

    def test_per_image_not_global(self):
        """Two images with different overlap; loss should reflect average."""
        criterion = DiceLoss()
        # Image 1: perfect
        img1_logit = torch.ones(1, 1, 32, 32) * 10.0
        img1_gt = torch.ones(1, 1, 32, 32)
        loss1 = criterion(img1_logit, img1_gt).item()

        # Image 2: worst case
        img2_logit = torch.ones(1, 1, 32, 32) * -10.0
        img2_gt = torch.ones(1, 1, 32, 32)
        loss2 = criterion(img2_logit, img2_gt).item()

        # Batch of both
        batch_logit = torch.cat([img1_logit, img2_logit], dim=0)
        batch_gt = torch.cat([img1_gt, img2_gt], dim=0)
        loss_batch = criterion(batch_logit, batch_gt).item()

        # Batch loss ≈ mean of individual losses (±numerical precision)
        expected = (loss1 + loss2) / 2.0
        assert loss_batch == pytest.approx(expected, abs=1e-4)

    def test_gradient_flows(self):
        criterion = DiceLoss()
        logits = torch.rand(2, 1, 32, 32, requires_grad=True)
        targets = (torch.rand(2, 1, 32, 32) > 0.5).float()
        loss = criterion(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()


class TestBCEWithLogitsLoss:
    def test_zero_for_perfect_prediction(self):
        criterion = BCEWithLogitsLoss()
        # BCE is ~0 for large positive logit with gt=1
        logits = torch.ones(2, 1, 64, 64) * 10.0
        targets = torch.ones(2, 1, 64, 64)
        loss = criterion(logits, targets)
        assert loss.item() < 0.01

    def test_gradient_flows(self):
        criterion = BCEWithLogitsLoss()
        logits = torch.rand(2, 1, 32, 32, requires_grad=True)
        targets = (torch.rand(2, 1, 32, 32) > 0.5).float()
        loss = criterion(logits, targets)
        loss.backward()
        assert logits.grad is not None


class TestCombinedLoss:
    def test_returns_scalar_and_components(self):
        criterion = CombinedLoss(alpha=0.5)
        logits = torch.rand(2, 1, 64, 64)
        targets = (torch.rand(2, 1, 64, 64) > 0.5).float()
        total, components = criterion(logits, targets)
        assert isinstance(total, torch.Tensor)
        assert total.ndim == 0  # scalar
        assert "dice_loss" in components
        assert "bce_loss" in components
        assert "total_loss" in components

    def test_alpha_weighting(self):
        """total = alpha * dice + (1-alpha) * bce."""
        criterion = CombinedLoss(alpha=0.5)
        logits = torch.ones(2, 1, 32, 32) * 2.0
        targets = torch.ones(2, 1, 32, 32)
        total, comps = criterion(logits, targets)
        expected = 0.5 * comps["dice_loss"] + 0.5 * comps["bce_loss"]
        assert comps["total_loss"] == pytest.approx(expected, abs=1e-5)

    def test_alpha_0_is_bce_only(self):
        criterion_bce = CombinedLoss(alpha=0.0)
        criterion_bce_ref = BCEWithLogitsLoss()
        logits = torch.rand(2, 1, 32, 32)
        targets = (torch.rand(2, 1, 32, 32) > 0.5).float()
        total_c, _ = criterion_bce(logits, targets)
        ref_bce = criterion_bce_ref(logits, targets)
        assert total_c.item() == pytest.approx(ref_bce.item(), abs=1e-5)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            CombinedLoss(alpha=1.5)
        with pytest.raises(ValueError):
            CombinedLoss(alpha=-0.1)

    def test_perfect_prediction_near_zero_loss(self):
        criterion = CombinedLoss(alpha=0.5)
        logits = torch.ones(4, 1, 64, 64) * 10.0
        targets = torch.ones(4, 1, 64, 64)
        total, _ = criterion(logits, targets)
        assert total.item() < 0.05


class TestBuildLoss:
    def test_combined(self):
        loss = build_loss("combined", alpha=0.5)
        assert isinstance(loss, CombinedLoss)

    def test_dice(self):
        loss = build_loss("dice")
        assert isinstance(loss, DiceLoss)

    def test_bce(self):
        loss = build_loss("bce")
        assert isinstance(loss, BCEWithLogitsLoss)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            build_loss("mse")
