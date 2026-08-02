"""
Illumination Model
====================
Solar illumination and shadow model for lunar south polar regions.
"""

from __future__ import annotations

import math

import numpy as np

from shared.constants import LUNAR


def compute_sun_elevation(
    time_hours: float,
    latitude_deg: float = -87.0,
    period_hours: float = LUNAR.synodic_period_hours,
) -> float:
    """Compute approximate sun elevation angle at a given time and latitude.

    Simplified model for south polar regions.

    Args:
        time_hours: Time since epoch (hours).
        latitude_deg: Latitude (degrees).
        period_hours: Lunar synodic period.

    Returns:
        Sun elevation angle in degrees (negative = below horizon).
    """
    phase = (time_hours % period_hours) / period_hours * 2 * math.pi
    max_elev = 90.0 + latitude_deg + LUNAR.obliquity_deg
    return max_elev * math.sin(phase)


def compute_illumination_map(
    dem: np.ndarray,
    sun_elevation_deg: float,
    sun_azimuth_deg: float = 180.0,
    pixel_size_m: float = 118.0,
) -> np.ndarray:
    """Compute binary illumination map using simple horizon model.

    A pixel is shadowed if the terrain between it and the sun direction
    blocks the line of sight (sun elevation below local horizon angle).

    Args:
        dem: 2D DEM elevation array.
        sun_elevation_deg: Sun elevation above horizon.
        sun_azimuth_deg: Sun azimuth (degrees from north).
        pixel_size_m: Pixel resolution.

    Returns:
        Binary illumination map (1=illuminated, 0=shadowed).
    """
    if sun_elevation_deg <= 0:
        return np.zeros_like(dem, dtype=np.uint8)

    illumination = np.ones_like(dem, dtype=np.uint8)
    h, w = dem.shape

    az_rad = math.radians(sun_azimuth_deg)
    elev_rad = math.radians(sun_elevation_deg)

    # Direction of shadow casting (opposite to sun)
    dy = -math.cos(az_rad)
    dx = -math.sin(az_rad)

    # Simplified: check a few steps in the shadow direction
    max_steps = min(h, w) // 2
    tan_elev = math.tan(elev_rad)

    for step in range(1, max_steps):
        offset_y = int(round(dy * step))
        offset_x = int(round(dx * step))
        distance_m = step * pixel_size_m

        # Height the sun would be at this distance
        sun_height = distance_m * tan_elev

        # For each pixel, check if terrain at offset blocks the sun
        for i in range(max(0, -offset_y), min(h, h - offset_y)):
            for j in range(max(0, -offset_x), min(w, w - offset_x)):
                blocker_elev = dem[i + offset_y, j + offset_x]
                target_elev = dem[i, j]
                if blocker_elev - target_elev > sun_height:
                    illumination[i, j] = 0

    return illumination


def compute_temperature(
    illuminated: bool,
    time_in_shadow_hours: float = 0.0,
    ambient_temp_k: float = 250.0,
    shadow_cooling_rate_kph: float = 5.0,
    psr_temp_k: float = LUNAR.psr_temp_k,
) -> float:
    """Estimate surface temperature based on illumination state.

    Args:
        illuminated: Whether the location is currently sunlit.
        time_in_shadow_hours: Hours spent in continuous shadow.
        ambient_temp_k: Temperature in sunlight.
        shadow_cooling_rate_kph: Cooling rate in shadow (K/hour).
        psr_temp_k: Minimum temperature (PSR floor).

    Returns:
        Estimated temperature in Kelvin.
    """
    if illuminated:
        return ambient_temp_k
    else:
        temp = ambient_temp_k - shadow_cooling_rate_kph * time_in_shadow_hours
        return max(temp, psr_temp_k)
