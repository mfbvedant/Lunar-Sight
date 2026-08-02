"""
Horn's Slope Algorithm
=======================
Vectorized NumPy implementation of Horn's 3×3 finite-difference algorithm
for computing terrain slope and aspect from a Digital Elevation Model.

Reference:
    Horn, B.K.P. (1981). "Hill shading and the reflectance map."
    Proceedings of the IEEE, 69(1), 14-47.
"""

from __future__ import annotations

import numpy as np


def horn_slope(
    dem: np.ndarray,
    dx: float,
    dy: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute slope and aspect from a DEM using Horn's algorithm.

    Horn's method uses a weighted 3×3 kernel that gives more weight to
    the center-adjacent pixels, providing better noise rejection than
    simple finite differences.

    The kernels for east-west and north-south gradients are::

        dz/dx kernel:         dz/dy kernel:
        [-1  0  1]            [ 1  2  1]
        [-2  0  2]            [ 0  0  0]
        [-1  0  1]            [-1 -2 -1]

    Args:
        dem: 2D array of elevation values (rows × cols).
            NaN values are handled via edge padding.
        dx: Pixel spacing in the east-west (X) direction, in meters.
        dy: Pixel spacing in the north-south (Y) direction, in meters.
            If None, uses dx (square pixels).

    Returns:
        Tuple of (slope, aspect):
            - slope: 2D array of slope angles in degrees [0°, 90°].
            - aspect: 2D array of aspect angles in degrees [0°, 360°),
              measured clockwise from north. Flat areas get aspect = -1.
    """
    if dy is None:
        dy = dx

    if dem.ndim != 2:
        raise ValueError(f"DEM must be 2D, got shape {dem.shape}")

    # Pad edges using reflection to handle boundaries
    padded = np.pad(dem.astype(np.float64), pad_width=1, mode='reflect')

    # Extract the 9 cells of the 3×3 neighborhood
    # Using the convention:
    #   a b c       (row-1, col-1) (row-1, col) (row-1, col+1)
    #   d e f   →   (row,   col-1) (row,   col) (row,   col+1)
    #   g h i       (row+1, col-1) (row+1, col) (row+1, col+1)
    a = padded[:-2, :-2]   # top-left
    b = padded[:-2, 1:-1]  # top-center
    c = padded[:-2, 2:]    # top-right
    d = padded[1:-1, :-2]  # middle-left
    # e = padded[1:-1, 1:-1]  # center (not used in gradient)
    f = padded[1:-1, 2:]   # middle-right
    g = padded[2:, :-2]    # bottom-left
    h = padded[2:, 1:-1]   # bottom-center
    i = padded[2:, 2:]     # bottom-right

    # Horn's weighted gradients
    dz_dx = ((c + 2.0 * f + i) - (a + 2.0 * d + g)) / (8.0 * dx)
    dz_dy = ((a + 2.0 * b + c) - (g + 2.0 * h + i)) / (8.0 * dy)

    # Slope magnitude (degrees)
    slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    slope_deg = np.degrees(slope_rad)

    # Aspect (degrees, clockwise from north)
    aspect_rad = np.arctan2(dz_dy, -dz_dx)
    aspect_deg = np.degrees(aspect_rad)

    # Convert from math convention to compass bearing:
    # Math: 0° = east, counter-clockwise positive
    # Compass: 0° = north, clockwise positive
    aspect_deg = (90.0 - aspect_deg) % 360.0

    # Mark flat areas (slope ≈ 0) with -1 aspect
    flat_mask = slope_deg < 1e-6
    aspect_deg[flat_mask] = -1.0

    return slope_deg, aspect_deg


def compute_slope_only(
    dem: np.ndarray,
    dx: float,
    dy: float | None = None,
) -> np.ndarray:
    """Compute slope only (faster if aspect is not needed).

    Args:
        dem: 2D elevation array.
        dx: Pixel spacing (meters) in X.
        dy: Pixel spacing (meters) in Y (default: same as dx).

    Returns:
        2D slope array in degrees.
    """
    slope, _ = horn_slope(dem, dx, dy)
    return slope


def compute_hillshade(
    dem: np.ndarray,
    dx: float,
    dy: float | None = None,
    azimuth_deg: float = 315.0,
    altitude_deg: float = 45.0,
) -> np.ndarray:
    """Compute analytical hillshade from DEM.

    Useful for visualization of terrain features.

    Args:
        dem: 2D elevation array.
        dx: Pixel spacing (meters) in X.
        dy: Pixel spacing (meters) in Y (default: same as dx).
        azimuth_deg: Sun azimuth in degrees (clockwise from north).
        altitude_deg: Sun altitude in degrees above horizon.

    Returns:
        2D hillshade array with values [0, 255].
    """
    slope_deg, aspect_deg = horn_slope(dem, dx, dy)

    slope_rad = np.radians(slope_deg)
    aspect_rad = np.radians(aspect_deg)
    azimuth_rad = np.radians(360.0 - azimuth_deg + 90.0)
    altitude_rad = np.radians(altitude_deg)

    # Handle flat areas (aspect = -1)
    flat_mask = aspect_deg < 0
    aspect_rad[flat_mask] = 0.0

    hillshade = (
        np.sin(altitude_rad) * np.cos(slope_rad)
        + np.cos(altitude_rad) * np.sin(slope_rad)
        * np.cos(azimuth_rad - aspect_rad)
    )

    # Scale to [0, 255]
    hillshade = np.clip(hillshade, 0, 1)
    hillshade = (hillshade * 255).astype(np.uint8)

    return hillshade
