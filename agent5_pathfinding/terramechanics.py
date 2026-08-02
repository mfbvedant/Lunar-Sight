"""
Terramechanics
===============
Mohr-Coulomb and Janosi-Hanamoto soil mechanics models for computing
wheel-soil interaction, slip ratios, and traversability on lunar regolith.

References:
    - Bekker, M.G. (1969). "Introduction to Terrain-Vehicle Systems."
    - Wong, J.Y. (2008). "Theory of Ground Vehicles."
"""

from __future__ import annotations

import math

import numpy as np

from shared.constants import REGOLITH_DEFAULTS, LUNAR


def mohr_coulomb(
    sigma: float,
    c: float = REGOLITH_DEFAULTS.cohesion_pa,
    phi_deg: float = REGOLITH_DEFAULTS.friction_angle_deg,
) -> float:
    """Compute maximum shear stress via Mohr-Coulomb failure criterion.

    τ_max = c + σ · tan(φ)

    Args:
        sigma: Normal stress (Pa).
        c: Soil cohesion (Pa).
        phi_deg: Internal friction angle (degrees).

    Returns:
        Maximum shear stress τ_max (Pa).
    """
    phi_rad = math.radians(phi_deg)
    return c + sigma * math.tan(phi_rad)


def janosi_hanamoto(
    sigma: float,
    j: float,
    c: float = REGOLITH_DEFAULTS.cohesion_pa,
    phi_deg: float = REGOLITH_DEFAULTS.friction_angle_deg,
    K: float = REGOLITH_DEFAULTS.shear_deformation_modulus_m,
) -> float:
    """Compute shear stress at displacement j via Janosi-Hanamoto model.

    τ(j) = τ_max · (1 - e^(-j/K))

    Where τ_max = c + σ·tan(φ) (Mohr-Coulomb).

    This models how shear stress builds up as the wheel displaces soil.
    At large j, τ → τ_max.

    Args:
        sigma: Normal stress (Pa).
        j: Shear displacement (m).
        c: Soil cohesion (Pa).
        phi_deg: Internal friction angle (degrees).
        K: Shear deformation modulus (m).

    Returns:
        Shear stress τ(j) (Pa).
    """
    tau_max = mohr_coulomb(sigma, c, phi_deg)
    return tau_max * (1.0 - math.exp(-j / K))


def compute_slip_ratio(
    slope_deg: float,
    mass_kg: float = 27.0,
    wheel_count: int = 6,
    wheel_radius_m: float = 0.105,
    wheel_width_m: float = 0.14,
    c: float = REGOLITH_DEFAULTS.cohesion_pa,
    phi_deg: float = REGOLITH_DEFAULTS.friction_angle_deg,
    K: float = REGOLITH_DEFAULTS.shear_deformation_modulus_m,
    gravity: float = LUNAR.surface_gravity_ms2,
) -> float:
    """Estimate wheel slip ratio on a given slope.

    Simplified model: slip increases with slope as the ratio of
    gravitational resistance to available traction.

    Args:
        slope_deg: Terrain slope in degrees.
        mass_kg: Rover mass.
        wheel_count: Number of wheels.
        wheel_radius_m: Wheel radius.
        wheel_width_m: Wheel width.
        c, phi_deg, K: Regolith parameters.
        gravity: Surface gravity.

    Returns:
        Slip ratio [0, 1]. Values near 1 mean near-total slipping.
    """
    slope_rad = math.radians(abs(slope_deg))

    # Weight component normal to slope
    weight = mass_kg * gravity
    normal_force = weight * math.cos(slope_rad) / wheel_count

    # Contact area per wheel (simplified)
    sinkage = 0.02  # Approximate sinkage (m)
    contact_length = 2.0 * math.sqrt(wheel_radius_m * sinkage)
    contact_area = contact_length * wheel_width_m

    # Normal stress
    sigma = normal_force / max(contact_area, 1e-6)

    # Available traction (Janosi-Hanamoto at full shear)
    tau_max = mohr_coulomb(sigma, c, phi_deg)
    traction = tau_max * contact_area * wheel_count

    # Resistance force (gravity along slope)
    resistance = weight * math.sin(slope_rad)

    # Slip ratio: how much traction is consumed by resistance
    if traction < 1e-6:
        return 1.0

    slip = resistance / traction
    return min(max(slip, 0.0), 1.0)


def compute_drawbar_pull(
    slope_deg: float,
    mass_kg: float = 27.0,
    wheel_count: int = 6,
    wheel_radius_m: float = 0.105,
    wheel_width_m: float = 0.14,
    c: float = REGOLITH_DEFAULTS.cohesion_pa,
    phi_deg: float = REGOLITH_DEFAULTS.friction_angle_deg,
    gravity: float = LUNAR.surface_gravity_ms2,
) -> float:
    """Compute net drawbar pull (force available for forward motion).

    Drawbar Pull = Traction - Resistance

    Args:
        slope_deg: Terrain slope.
        mass_kg: Rover mass.
        Other args: physical parameters.

    Returns:
        Drawbar pull in Newtons. Negative means rover cannot climb.
    """
    slope_rad = math.radians(abs(slope_deg))
    weight = mass_kg * gravity
    normal_force = weight * math.cos(slope_rad) / wheel_count

    sinkage = 0.02
    contact_length = 2.0 * math.sqrt(wheel_radius_m * sinkage)
    contact_area = contact_length * wheel_width_m

    sigma = normal_force / max(contact_area, 1e-6)
    tau_max = mohr_coulomb(sigma, c, phi_deg)
    traction = tau_max * contact_area * wheel_count

    resistance = weight * math.sin(slope_rad)

    return traction - resistance


def is_traversable(
    slope_deg: float,
    max_slope_deg: float = 10.0,
    mass_kg: float = 27.0,
    **kwargs,
) -> bool:
    """Check if terrain at given slope is traversable.

    Traversability requires:
        1. Slope < max_slope_deg (mechanical limit)
        2. Positive drawbar pull (traction > resistance)
        3. Slip ratio < 0.8 (practical mobility limit)

    Args:
        slope_deg: Terrain slope.
        max_slope_deg: Maximum traversable slope.
        mass_kg: Rover mass.

    Returns:
        True if traversable.
    """
    if abs(slope_deg) >= max_slope_deg:
        return False

    dbp = compute_drawbar_pull(slope_deg, mass_kg=mass_kg, **kwargs)
    if dbp <= 0:
        return False

    slip = compute_slip_ratio(slope_deg, mass_kg=mass_kg, **kwargs)
    if slip >= 0.8:
        return False

    return True
