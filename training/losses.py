"""
training/losses.py

Loss functions for segmentation training.

Implements:
  - DiceLoss: computed per image, then batch-averaged (audit fix LOW-23)
  - BCEWithLogitsLoss: pixel-level binary cross-entropy
  - CombinedLoss: weighted sum with separate component logging

Design:
  Per-image Dice (not global pixel-level Dice) is used to match evaluation
  protocol and avoid bias from large background regions.

Audit fix: CRITICAL-13 — separate logging of Dice and BCE components.
Audit fix: LOW-23 — per-image computation before batch average.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.

    Computes Dice per image in the batch, then averages.
    This matches the evaluation protocol (per-image Dice → mean).

    Args:
        smooth: Laplace smoothing constant to prevent division by zero.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute mean Dice loss over a batch.

        Args:
            logits: (B, 1, H, W) raw model output (before sigmoid).
            targets: (B, 1, H, W) binary ground truth in [0, 1].

        Returns:
            Scalar Dice loss (1 - mean Dice score).
        """
        probs = torch.sigmoid(logits)

        # Flatten per image: (B, H*W)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        # Per-image intersection and union
        intersection = (probs_flat * targets_flat).sum(dim=1)
        sum_pred = probs_flat.sum(dim=1)
        sum_gt = targets_flat.sum(dim=1)

        # Per-image Dice
        dice_per_image = (2.0 * intersection + self.smooth) / (
            sum_pred + sum_gt + self.smooth
        )

        return 1.0 - dice_per_image.mean()


class BCEWithLogitsLoss(nn.Module):
    """
    Binary cross-entropy with logits, averaged over pixels and batch.

    Thin wrapper around torch's BCEWithLogitsLoss with consistent interface.

    Args:
        pos_weight: Optional positive class weight for imbalanced data.
    """

    def __init__(self, pos_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self._loss = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight, reduction="mean"
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, H, W) raw model output.
            targets: (B, 1, H, W) binary ground truth in [0, 1].

        Returns:
            Scalar BCE loss.
        """
        return self._loss(logits, targets.float())


class CombinedLoss(nn.Module):
    """
    Combined Dice + BCE loss.

    Final loss = alpha * Dice + (1 - alpha) * BCE.

    Logs both components separately so training curves can be inspected.

    Audit fix: CRITICAL-13 — separate component logging.

    Args:
        alpha: Weight for Dice component (0.5 = equal weighting).
        smooth: Smoothing constant for DiceLoss.
    """

    def __init__(self, alpha: float = 0.5, smooth: float = 1.0) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss = BCEWithLogitsLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss and return components.

        Args:
            logits: (B, 1, H, W) model output.
            targets: (B, 1, H, W) binary ground truth.

        Returns:
            total_loss: Scalar combined loss.
            components: Dict with 'dice_loss', 'bce_loss', 'total_loss'.
        """
        dice = self.dice_loss(logits, targets)
        bce = self.bce_loss(logits, targets)
        total = self.alpha * dice + (1.0 - self.alpha) * bce

        components = {
            "dice_loss": dice.item(),
            "bce_loss": bce.item(),
            "total_loss": total.item(),
        }

        return total, components


def build_loss(
    loss_name: str = "combined",
    alpha: float = 0.5,
) -> nn.Module:
    """
    Factory function for loss modules.

    Args:
        loss_name: One of 'combined', 'dice', 'bce'.
        alpha: Weight for Dice in combined loss.

    Returns:
        Instantiated loss module.

    Raises:
        ValueError: If loss_name is not recognized.
    """
    if loss_name == "combined":
        return CombinedLoss(alpha=alpha)
    elif loss_name == "dice":
        return DiceLoss()
    elif loss_name == "bce":
        return BCEWithLogitsLoss()
    else:
        raise ValueError(
            f"Unknown loss: '{loss_name}'. Choose: combined, dice, bce."
        )
