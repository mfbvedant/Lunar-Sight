"""
LunarSight Geospatial Utilities
================================
Coordinate reference system transforms, bounding box operations,
and geospatial helper functions for lunar polar stereographic mapping.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from shared.constants import LUNAR

logger = logging.getLogger(__name__)


# ============================================================
# Bounding Box
# ============================================================

@dataclass
class BoundingBox:
    """Geographic bounding box in latitude/longitude.

    Attributes:
        lat_min: Southern latitude boundary (degrees).
        lat_max: Northern latitude boundary (degrees).
        lon_min: Western longitude boundary (degrees).
        lon_max: Eastern longitude boundary (degrees).
    """

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def __post_init__(self):
        if self.lat_min >= self.lat_max:
            raise ValueError(
                f"lat_min ({self.lat_min}) must be < lat_max ({self.lat_max})"
            )
        if self.lon_min >= self.lon_max:
            raise ValueError(
                f"lon_min ({self.lon_min}) must be < lon_max ({self.lon_max})"
            )

    @property
    def center_lat(self) -> float:
        return (self.lat_min + self.lat_max) / 2.0

    @property
    def center_lon(self) -> float:
        return (self.lon_min + self.lon_max) / 2.0

    @property
    def width_deg(self) -> float:
        return self.lon_max - self.lon_min

    @property
    def height_deg(self) -> float:
        return self.lat_max - self.lat_min

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.lat_min, self.lat_max, self.lon_min, self.lon_max)

    def width_m(self, radius_m: float = LUNAR.radius_m) -> float:
        """Approximate east-west extent in meters at center latitude."""
        lat_rad = math.radians(self.center_lat)
        return math.radians(self.width_deg) * radius_m * math.cos(lat_rad)

    def height_m(self, radius_m: float = LUNAR.radius_m) -> float:
        """Approximate north-south extent in meters."""
        return math.radians(self.height_deg) * radius_m

    @classmethod
    def from_config(cls, config: dict) -> "BoundingBox":
        """Create from a mission_config.yaml ``target.bbox`` dict."""
        bbox = config["target"]["bbox"]
        return cls(
            lat_min=bbox["lat_min"],
            lat_max=bbox["lat_max"],
            lon_min=bbox["lon_min"],
            lon_max=bbox["lon_max"],
        )

    def contains(self, lat: float, lon: float) -> bool:
        """Check if a point is inside this bounding box."""
        return (
            self.lat_min <= lat <= self.lat_max
            and self.lon_min <= lon <= self.lon_max
        )


# ============================================================
# Lunar Polar Stereographic Projection
# ============================================================

def lunar_polar_stereo_proj4(
    hemisphere: str = "south",
    radius_m: float = LUNAR.radius_m,
) -> str:
    """Generate a PROJ4 string for Lunar Polar Stereographic projection.

    Args:
        hemisphere: "south" or "north".
        radius_m: Lunar radius in meters.

    Returns:
        PROJ4 projection string.
    """
    lat_0 = -90.0 if hemisphere == "south" else 90.0
    return (
        f"+proj=stere +lat_0={lat_0} +lon_0=0 "
        f"+k=1 +x_0=0 +y_0=0 "
        f"+a={radius_m} +b={radius_m} +units=m +no_defs"
    )


def latlon_to_stereo(
    lat: np.ndarray | float,
    lon: np.ndarray | float,
    hemisphere: str = "south",
    radius_m: float = LUNAR.radius_m,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Convert geographic (lat, lon) to polar stereographic (x, y).

    Pure NumPy implementation — no GDAL/pyproj dependency.

    Args:
        lat: Latitude(s) in degrees.
        lon: Longitude(s) in degrees.
        hemisphere: "south" or "north".
        radius_m: Body radius in meters.

    Returns:
        (x, y) in meters on the stereographic plane.
    """
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)

    if hemisphere == "south":
        # South polar stereographic: project from North Pole
        k = 2.0 * radius_m / (1.0 - np.sin(lat_r))
        x = k * np.cos(lat_r) * np.sin(lon_r)
        y = k * np.cos(lat_r) * np.cos(lon_r)
    else:
        # North polar stereographic: project from South Pole
        k = 2.0 * radius_m / (1.0 + np.sin(lat_r))
        x = k * np.cos(lat_r) * np.sin(lon_r)
        y = -k * np.cos(lat_r) * np.cos(lon_r)

    return x, y


def stereo_to_latlon(
    x: np.ndarray | float,
    y: np.ndarray | float,
    hemisphere: str = "south",
    radius_m: float = LUNAR.radius_m,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Convert polar stereographic (x, y) back to geographic (lat, lon).

    Args:
        x: Easting(s) in meters.
        y: Northing(s) in meters.
        hemisphere: "south" or "north".
        radius_m: Body radius in meters.

    Returns:
        (lat, lon) in degrees.
    """
    rho = np.sqrt(x ** 2 + y ** 2)
    c = 2.0 * np.arctan2(rho, 2.0 * radius_m)

    if hemisphere == "south":
        lat = np.degrees(np.arcsin(-np.cos(c)))
        lon = np.degrees(np.arctan2(x, y))
    else:
        lat = np.degrees(np.arcsin(np.cos(c)))
        lon = np.degrees(np.arctan2(x, -y))

    return lat, lon


# ============================================================
# Pixel / Resolution Helpers
# ============================================================

def compute_pixel_size_m(
    bbox: BoundingBox,
    raster_shape: tuple[int, int],
) -> tuple[float, float]:
    """Compute approximate pixel size in meters.

    Args:
        bbox: Geographic bounding box.
        raster_shape: (rows, cols) of the raster.

    Returns:
        (dy_m, dx_m) — pixel size in north-south and east-west directions.
    """
    dy_m = bbox.height_m() / raster_shape[0]
    dx_m = bbox.width_m() / raster_shape[1]
    return dy_m, dx_m


def create_coordinate_grids(
    bbox: BoundingBox,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Create latitude and longitude grids for a raster.

    Args:
        bbox: Geographic bounding box.
        shape: (rows, cols) of the output grid.

    Returns:
        (lat_grid, lon_grid) — 2D arrays of coordinates.
    """
    lats = np.linspace(bbox.lat_max, bbox.lat_min, shape[0])
    lons = np.linspace(bbox.lon_min, bbox.lon_max, shape[1])
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    return lat_grid, lon_grid
