"""
Pseudo Label Generation
========================
Generate sparse seed labels for weakly-supervised segmentation from
polarimetric feature maps using physics-based thresholds.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from shared.constants import EPS

logger = logging.getLogger(__name__)


def generate_pseudo_labels(
    l_cpr: np.ndarray,
    s_cpr: Optional[np.ndarray],
    dop: np.ndarray,
    ice_cpr_min: float = 1.0,
    ice_dop_max: float = 0.13,
    s_cpr_min: float = 1.0,
    rock_m: Optional[np.ndarray] = None,
    rock_m_min: float = 0.7,
    odd_bounce: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Generate sparse pseudo-labels from polarimetric criteria.

    Label values:
        - 1 (ICE): L-band CPR > threshold AND S-band CPR > threshold AND DOP < threshold
        - 0 (ROCK): sunlit/equatorial pixels with high odd-bounce scattering
        - NaN (UNLABELED): everything else

    Args:
        l_cpr: L-band CPR array (H×W).
        s_cpr: S-band CPR array (H×W), optional.
        dop: Degree of polarization (H×W).
        ice_cpr_min: CPR threshold for ice candidates.
        ice_dop_max: DOP threshold for ice candidates.
        s_cpr_min: S-band CPR threshold.
        rock_m: Degree of polarization for rock classification.
        rock_m_min: Threshold for rock m.
        odd_bounce: Binary odd-bounce dominance mask, optional.

    Returns:
        Tuple of:
            - label_map: (H×W) with 0=rock, 1=ice, NaN=unlabeled.
            - stats: Dict with seed statistics.
    """
    h, w = l_cpr.shape
    label_map = np.full((h, w), np.nan, dtype=np.float32)

    # ---- Ice seeds ----
    ice_mask = (l_cpr > ice_cpr_min) & (dop < ice_dop_max)

    if s_cpr is not None:
        ice_mask = ice_mask & (s_cpr > s_cpr_min)

    # Filter NaN regions
    ice_mask = ice_mask & np.isfinite(l_cpr) & np.isfinite(dop)
    label_map[ice_mask] = 1.0

    # ---- Rock seeds ----
    if rock_m is not None:
        rock_mask = rock_m > rock_m_min
        if odd_bounce is not None:
            rock_mask = rock_mask & odd_bounce.astype(bool)
        rock_mask = rock_mask & np.isfinite(rock_m)
        # Don't overwrite ice seeds
        rock_mask = rock_mask & ~ice_mask
        label_map[rock_mask] = 0.0
    else:
        rock_mask = np.zeros((h, w), dtype=bool)

    # ---- Statistics ----
    total = h * w
    valid = np.isfinite(l_cpr).sum()
    n_ice = ice_mask.sum()
    n_rock = rock_mask.sum()
    n_unlabeled = total - n_ice - n_rock

    stats = {
        'total_pixels': int(total),
        'valid_pixels': int(valid),
        'ice_seeds': int(n_ice),
        'rock_seeds': int(n_rock),
        'unlabeled': int(n_unlabeled),
        'pct_ice': float(n_ice / max(valid, 1) * 100),
        'pct_rock': float(n_rock / max(valid, 1) * 100),
        'pct_unlabeled': float(n_unlabeled / max(valid, 1) * 100),
    }

    logger.info(
        f"Pseudo-labels: {n_ice} ice ({stats['pct_ice']:.2f}%), "
        f"{n_rock} rock ({stats['pct_rock']:.2f}%), "
        f"{n_unlabeled} unlabeled ({stats['pct_unlabeled']:.2f}%)"
    )

    return label_map, stats


def expand_pseudo_labels(
    label_map: np.ndarray,
    predictions: np.ndarray,
    confidence: np.ndarray,
    confidence_threshold: float = 0.95,
) -> tuple[np.ndarray, dict[str, int]]:
    """Expand pseudo-labels with high-confidence model predictions.

    Used in self-training iterations: the model's own high-confidence
    predictions on unlabeled pixels become new training labels.

    Args:
        label_map: Current label map (H×W) with NaN for unlabeled.
        predictions: Model predictions (H×W), 0=rock, 1=ice.
        confidence: Confidence scores (H×W), [0, 1].
        confidence_threshold: Minimum confidence to accept a prediction.

    Returns:
        Tuple of (expanded_label_map, expansion_stats).
    """
    expanded = label_map.copy()

    # Find high-confidence unlabeled pixels
    unlabeled = np.isnan(label_map)
    high_conf = confidence > confidence_threshold

    new_labels = unlabeled & high_conf
    expanded[new_labels] = predictions[new_labels].astype(np.float32)

    new_ice = new_labels & (predictions == 1)
    new_rock = new_labels & (predictions == 0)

    stats = {
        'new_labels_total': int(new_labels.sum()),
        'new_ice': int(new_ice.sum()),
        'new_rock': int(new_rock.sum()),
        'remaining_unlabeled': int(np.isnan(expanded).sum()),
    }

    logger.info(
        f"Label expansion: +{stats['new_labels_total']} labels "
        f"(+{stats['new_ice']} ice, +{stats['new_rock']} rock), "
        f"{stats['remaining_unlabeled']} still unlabeled"
    )

    return expanded, stats
