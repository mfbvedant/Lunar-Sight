"""
Graph Builder
==============
Convert DEM + slope arrays into a searchable graph for pathfinding.
"""

from __future__ import annotations

import logging

import numpy as np

from agent5_pathfinding.terramechanics import is_traversable

logger = logging.getLogger(__name__)


def build_traversability_mask(
    slope: np.ndarray,
    max_slope_deg: float = 10.0,
    mass_kg: float = 27.0,
) -> np.ndarray:
    """Create binary traversability mask from slope array.

    Args:
        slope: 2D slope array (degrees).
        max_slope_deg: Maximum traversable slope.
        mass_kg: Rover mass.

    Returns:
        Binary mask (True=traversable).
    """
    mask = np.zeros_like(slope, dtype=bool)

    for i in range(slope.shape[0]):
        for j in range(slope.shape[1]):
            if np.isfinite(slope[i, j]):
                mask[i, j] = is_traversable(slope[i, j], max_slope_deg, mass_kg)

    pct = mask.sum() / mask.size * 100
    logger.info(f"Traversability: {mask.sum()}/{mask.size} pixels ({pct:.1f}%)")

    return mask


def find_best_ice_target(
    ice_mask: np.ndarray,
    confidence: np.ndarray,
    traversable: np.ndarray,
    start: tuple[int, int],
    max_distance_pixels: int = 500,
) -> tuple[int, int] | None:
    """Find the highest-confidence reachable ice deposit.

    Selects the ice pixel with highest confidence that is:
        1. Within max_distance_pixels of the start
        2. Has traversable terrain nearby

    Args:
        ice_mask: Binary ice mask.
        confidence: Confidence map [0, 1].
        traversable: Traversability mask.
        start: Start position (row, col).
        max_distance_pixels: Maximum search radius.

    Returns:
        (row, col) of best target, or None.
    """
    ice_pixels = np.argwhere(ice_mask > 0)

    if len(ice_pixels) == 0:
        logger.warning("No ice pixels found")
        return None

    # Filter by distance
    distances = np.sqrt(
        (ice_pixels[:, 0] - start[0]) ** 2 +
        (ice_pixels[:, 1] - start[1]) ** 2
    )
    reachable = distances < max_distance_pixels

    if not np.any(reachable):
        logger.warning(f"No ice within {max_distance_pixels} pixels of start")
        return None

    # Filter by confidence
    candidates = ice_pixels[reachable]
    confidences = confidence[candidates[:, 0], candidates[:, 1]]

    # Sort by confidence (descending)
    best_idx = np.argmax(confidences)
    target = tuple(candidates[best_idx])

    logger.info(
        f"Best ice target: ({target[0]}, {target[1]}), "
        f"confidence={confidences[best_idx]:.3f}, "
        f"distance={distances[reachable][best_idx]:.0f} pixels"
    )

    return target


def subsample_grid(
    slope: np.ndarray,
    factor: int = 2,
) -> np.ndarray:
    """Subsample the slope grid for faster initial planning.

    Uses max-slope within each block to be conservative.

    Args:
        slope: Full-resolution slope array.
        factor: Subsampling factor.

    Returns:
        Subsampled slope array.
    """
    h, w = slope.shape
    new_h = h // factor
    new_w = w // factor

    subsampled = np.zeros((new_h, new_w))
    for i in range(new_h):
        for j in range(new_w):
            block = slope[i*factor:(i+1)*factor, j*factor:(j+1)*factor]
            valid = block[np.isfinite(block)]
            subsampled[i, j] = np.max(valid) if len(valid) > 0 else np.nan

    return subsampled
