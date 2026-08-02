"""
Slip-Aware Cost Function
==========================
Edge cost computation for the A* pathfinder incorporating terrain slope,
wheel slip, and energy consumption.
"""

from __future__ import annotations

import math

import numpy as np

from agent5_pathfinding.terramechanics import compute_slip_ratio, is_traversable


def compute_edge_cost(
    slope_deg: float,
    distance_m: float,
    max_slope_deg: float = 10.0,
    mass_kg: float = 27.0,
    power_nominal_w: float = 15.0,
    velocity_ms: float = 0.01,
    **terra_kwargs,
) -> tuple[float, float, float]:
    """Compute the cost of traversing an edge (pixel-to-pixel).

    Cost components:
        1. Base: Euclidean distance
        2. Slip penalty: distance / (1 - slip) — more distance needed at high slip
        3. Energy: power × time (higher power at high slip)

    Args:
        slope_deg: Slope at the target pixel.
        distance_m: Euclidean distance between nodes.
        max_slope_deg: Rover maximum traversable slope.
        mass_kg: Rover mass.
        power_nominal_w: Nominal power draw.
        velocity_ms: Nominal velocity.

    Returns:
        Tuple of (cost, energy_wh, time_s).
        cost = inf if not traversable.
    """
    # Check traversability
    if not is_traversable(slope_deg, max_slope_deg, mass_kg, **terra_kwargs):
        return float('inf'), float('inf'), float('inf')

    # Slip ratio
    slip = compute_slip_ratio(slope_deg, mass_kg=mass_kg, **terra_kwargs)

    # Effective distance (slip increases actual distance)
    effective_distance = distance_m / max(1.0 - slip, 0.01)

    # Effective velocity (slower at high slip)
    effective_velocity = velocity_ms * (1.0 - slip)
    effective_velocity = max(effective_velocity, 1e-4)

    # Time to traverse
    time_s = effective_distance / effective_velocity

    # Power increases with slip (motor works harder)
    power_draw = power_nominal_w * (1.0 + slip * 2.0)
    energy_wh = power_draw * time_s / 3600.0

    # Cost = base distance + slip penalty + slope penalty
    slope_penalty = 1.0 + (slope_deg / max_slope_deg) ** 2
    cost = effective_distance * slope_penalty

    return cost, energy_wh, time_s


def compute_cost_map(
    slope: np.ndarray,
    pixel_size_m: float,
    max_slope_deg: float = 10.0,
    mass_kg: float = 27.0,
) -> np.ndarray:
    """Compute a traversal cost map from a slope array.

    Args:
        slope: 2D slope array (degrees).
        pixel_size_m: Pixel resolution in meters.
        max_slope_deg: Max traversable slope.
        mass_kg: Rover mass.

    Returns:
        2D cost array. inf for impassable pixels.
    """
    cost_map = np.full_like(slope, np.nan, dtype=np.float64)

    for i in range(slope.shape[0]):
        for j in range(slope.shape[1]):
            s = slope[i, j]
            if np.isnan(s):
                cost_map[i, j] = float('inf')
            else:
                c, _, _ = compute_edge_cost(
                    s, pixel_size_m, max_slope_deg, mass_kg
                )
                cost_map[i, j] = c

    return cost_map
