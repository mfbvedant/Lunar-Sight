"""
Complex Despeckling Loss Functions
====================================
Complex Frobenius loss + Phase Coherence loss for training the CV-CNN.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ComplexFrobeniusLoss(nn.Module):
    """Complex Frobenius norm loss: ‖C_clean - C_pred‖_F in complex domain.

    Measures the element-wise distance between target and predicted complex
    covariance matrices.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        # Frobenius norm: sqrt(sum(|z|²)) → we minimize sum(|z|²) for stability
        frob = torch.mean(torch.abs(diff) ** 2)
        return frob


class PhaseCoherenceLoss(nn.Module):
    """Phase coherence loss: penalizes degradation of off-diagonal phase angles.

    L_phase = 1 - cos(∠C_target_offdiag - ∠C_pred_offdiag)

    This ensures the network preserves the relative phase between
    polarization channels, which is critical for polarimetric analysis.
    """

    def __init__(self, off_diag_indices: list[int] | None = None):
        """
        Args:
            off_diag_indices: Channel indices corresponding to off-diagonal
                (complex) elements. For C₂ packed tensor: [1, 2] (Re, Im of C₁₂).
                If None, uses all channels.
        """
        super().__init__()
        self.off_diag_indices = off_diag_indices

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.off_diag_indices is not None:
            # Select off-diagonal channels
            pred_off = pred[:, self.off_diag_indices]
            target_off = target[:, self.off_diag_indices]
        else:
            pred_off = pred
            target_off = target

        # Compute phase angles
        pred_phase = torch.angle(pred_off)
        target_phase = torch.angle(target_off)

        # Phase difference loss: 1 - cos(Δphase)
        phase_diff = target_phase - pred_phase
        loss = 1.0 - torch.cos(phase_diff)

        return torch.mean(loss)


class CombinedDespecklingLoss(nn.Module):
    """Combined loss: α·L_frobenius + β·L_phase.

    Args:
        alpha: Weight for Frobenius loss.
        beta: Weight for phase coherence loss.
        off_diag_indices: Channel indices for off-diagonal elements.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        off_diag_indices: list[int] | None = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.frobenius = ComplexFrobeniusLoss()
        self.phase = PhaseCoherenceLoss(off_diag_indices)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined loss.

        Returns:
            Tuple of (total_loss, component_dict).
        """
        l_frob = self.frobenius(pred, target)
        l_phase = self.phase(pred, target)

        total = self.alpha * l_frob + self.beta * l_phase

        components = {
            'total': total.item(),
            'frobenius': l_frob.item(),
            'phase': l_phase.item(),
        }

        return total, components
