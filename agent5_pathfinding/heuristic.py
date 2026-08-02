"""
A* Heuristic
==============
Admissible heuristic for the kinodynamic A* pathfinder with battery
and thermal depletion awareness.
"""

from __future__ import annotations

import math

import numpy as np


def euclidean_heuristic(
    current: tuple[int, int],
    goal: tuple[int, int],
    pixel_size_m: float = 118.0,
) -> float:
    """Euclidean distance heuristic (admissible).

    Args:
        current: (row, col) of current node.
        goal: (row, col) of goal node.
        pixel_size_m: Pixel size in meters.

    Returns:
        Distance in meters (never overestimates).
    """
    dy = (current[0] - goal[0]) * pixel_size_m
    dx = (current[1] - goal[1]) * pixel_size_m
    return math.sqrt(dy ** 2 + dx ** 2)


def battery_aware_heuristic(
    current: tuple[int, int],
    goal: tuple[int, int],
    battery_remaining_wh: float,
    pixel_size_m: float = 118.0,
    power_nominal_w: float = 15.0,
    velocity_ms: float = 0.01,
    return_trip: bool = True,
    start: tuple[int, int] | None = None,
) -> float:
    """Heuristic incorporating battery depletion estimate.

    Adds a penalty if the remaining battery may be insufficient for:
        1. Reaching the goal
        2. Returning to the start (if return_trip=True)

    Args:
        current: Current position (row, col).
        goal: Goal position (row, col).
        battery_remaining_wh: Remaining battery energy (Wh).
        pixel_size_m: Pixel size in meters.
        power_nominal_w: Nominal power draw.
        velocity_ms: Nominal velocity.
        return_trip: If True, account for return trip energy.
        start: Start position for return trip estimation.

    Returns:
        Heuristic cost (admissible).
    """
    # Distance to goal
    dist_to_goal = euclidean_heuristic(current, goal, pixel_size_m)

    # Estimated energy to reach goal
    time_to_goal = dist_to_goal / max(velocity_ms, 1e-6)
    energy_to_goal = power_nominal_w * time_to_goal / 3600.0

    # Return trip energy estimate
    return_energy = 0.0
    if return_trip and start is not None:
        dist_return = euclidean_heuristic(goal, start, pixel_size_m)
        time_return = dist_return / max(velocity_ms, 1e-6)
        return_energy = power_nominal_w * time_return / 3600.0

    total_energy_needed = energy_to_goal + return_energy

    if total_energy_needed > battery_remaining_wh:
        # Battery insufficient — add large penalty
        return dist_to_goal + 1e6
    else:
        return dist_to_goal


def thermal_aware_heuristic(
    current: tuple[int, int],
    goal: tuple[int, int],
    current_temp_k: float,
    thermal_min_k: float = 173.0,
    pixel_size_m: float = 118.0,
    shadow_cooling_rate: float = 5.0,
    velocity_ms: float = 0.01,
) -> float:
    """Heuristic with thermal budget awareness.

    Adds a penalty if the rover may cool below minimum operating
    temperature before reaching the goal.

    Args:
        current: Current position.
        goal: Goal position.
        current_temp_k: Current rover temperature (K).
        thermal_min_k: Minimum operating temperature.
        pixel_size_m: Pixel resolution.
        shadow_cooling_rate: Cooling rate in shadow (K/hour).
        velocity_ms: Rover velocity.

    Returns:
        Heuristic cost (admissible).
    """
    dist = euclidean_heuristic(current, goal, pixel_size_m)
    time_hours = dist / max(velocity_ms, 1e-6) / 3600.0

    # Worst case: all in shadow
    projected_temp = current_temp_k - shadow_cooling_rate * time_hours

    if projected_temp < thermal_min_k:
        return dist + 1e6  # Too cold
    else:
        return dist


def combined_heuristic(
    current: tuple[int, int],
    goal: tuple[int, int],
    battery_wh: float,
    temp_k: float,
    pixel_size_m: float = 118.0,
    start: tuple[int, int] | None = None,
    **kwargs,
) -> float:
    """Combined heuristic using max of all component heuristics.

    Taking the max preserves admissibility while being more informed.
    """
    h_dist = euclidean_heuristic(current, goal, pixel_size_m)
    h_battery = battery_aware_heuristic(
        current, goal, battery_wh, pixel_size_m,
        start=start, **kwargs,
    )
    h_thermal = thermal_aware_heuristic(
        current, goal, temp_k, pixel_size_m=pixel_size_m, **kwargs,
    )

    return max(h_dist, h_battery, h_thermal)
