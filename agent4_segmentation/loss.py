"""
Segmentation Loss Functions
==============================
NaN-masked Binary Cross-Entropy + Dice loss for weakly-supervised training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NaNMaskedBCELoss(nn.Module):
    """Binary cross-entropy that ignores NaN/unlabeled pixels.

    Only computes loss on pixels where valid_mask == True.
    Supports class weighting to balance ice (minority) vs rock (majority).
    """

    def __init__(self, class_weights: list[float] | None = None):
        super().__init__()
        self.class_weights = class_weights

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: Raw model output (B, C, H, W).
            targets: Ground truth labels (B, H, W), long.
            valid_mask: (B, H, W) float mask, 1.0 where valid.
        """
        # Convert targets to one-hot for multi-class BCE
        n_classes = logits.shape[1]

        if n_classes == 1:
            # Binary case
            logits_flat = logits.squeeze(1)
            targets_float = targets.float()
            bce = F.binary_cross_entropy_with_logits(
                logits_flat, targets_float, reduction='none'
            )
        else:
            # Multi-class: compute cross-entropy
            bce = F.cross_entropy(logits, targets, reduction='none')

        # Apply class weights
        if self.class_weights is not None and n_classes > 1:
            weights = torch.tensor(self.class_weights, device=logits.device)
            weight_map = weights[targets]
            bce = bce * weight_map

        # Apply valid mask
        masked_bce = bce * valid_mask
        n_valid = valid_mask.sum().clamp(min=1.0)

        return masked_bce.sum() / n_valid


class NaNMaskedDiceLoss(nn.Module):
    """Dice loss that ignores NaN/unlabeled pixels.

    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Loss = 1 - Dice
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        n_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)

        dice_sum = 0.0
        for c in range(n_classes):
            pred_c = probs[:, c] * valid_mask
            target_c = (targets == c).float() * valid_mask

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_sum += dice

        return 1.0 - dice_sum / n_classes


class CombinedSegmentationLoss(nn.Module):
    """Combined loss: α·BCE_masked + β·Dice_masked.

    Args:
        alpha: BCE weight.
        beta: Dice weight.
        class_weights: Optional class weights for BCE.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        class_weights: list[float] | None = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce = NaNMaskedBCELoss(class_weights)
        self.dice = NaNMaskedDiceLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        l_bce = self.bce(logits, targets, valid_mask)
        l_dice = self.dice(logits, targets, valid_mask)

        total = self.alpha * l_bce + self.beta * l_dice

        return total, {
            'total': total.item(),
            'bce': l_bce.item(),
            'dice': l_dice.item(),
        }
