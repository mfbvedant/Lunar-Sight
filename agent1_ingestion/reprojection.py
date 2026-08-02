"""
Reprojection
==============
Reproject rasters into Lunar Polar Stereographic CRS using GDAL/Rasterio,
ensuring pixel alignment across all data sources.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from shared.constants import LUNAR
from shared.geo_utils import lunar_polar_stereo_proj4, BoundingBox

logger = logging.getLogger(__name__)


def reproject_to_polar_stereographic(
    input_path: str | Path,
    output_path: str | Path,
    target_resolution_m: Optional[float] = None,
    hemisphere: str = "south",
    resampling: str = "bilinear",
    reference_path: Optional[str | Path] = None,
) -> Path:
    """Reproject a raster to Lunar Polar Stereographic CRS.

    Uses rasterio (with GDAL backend) to warp the raster into a polar
    stereographic projection centered on the lunar south (or north) pole.

    Args:
        input_path: Path to the input raster (GeoTIFF, IMG, etc.).
        output_path: Path for the output reprojected raster.
        target_resolution_m: Desired pixel size in meters. If None, inferred
            from the reference or input raster.
        hemisphere: "south" or "north".
        resampling: Resampling algorithm — "bilinear", "nearest", "cubic".
        reference_path: Optional path to a reference raster for grid alignment.

    Returns:
        Path to the reprojected output raster.
    """
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject
    from rasterio.enums import Resampling

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Target CRS as PROJ4 string
    dst_crs = lunar_polar_stereo_proj4(hemisphere, LUNAR.radius_m)

    # Map resampling string to enum
    resamp_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "lanczos": Resampling.lanczos,
    }
    resamp = resamp_map.get(resampling, Resampling.bilinear)

    with rasterio.open(str(input_path)) as src:
        # If we have a reference raster, match its grid
        if reference_path and Path(reference_path).exists():
            return _reproject_to_match(
                src, output_path, reference_path, dst_crs, resamp
            )

        # Calculate default transform
        transform, width, height = calculate_default_transform(
            src.crs or dst_crs,  # Input CRS (use dst if input has none)
            dst_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=target_resolution_m,
        )

        # Build output profile
        profile = src.profile.copy()
        profile.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'driver': 'GTiff',
            'compress': 'lzw',
        })

        with rasterio.open(str(output_path), 'w', **profile) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs or dst_crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resamp,
                )

    logger.info(
        f"Reprojected {input_path.name} → {output_path.name} "
        f"({width}×{height}, {resampling})"
    )
    return output_path


def _reproject_to_match(
    src,
    output_path: Path,
    reference_path: str | Path,
    dst_crs: str,
    resampling,
) -> Path:
    """Reproject source raster to match a reference raster's grid exactly.

    Ensures pixel alignment by using the reference's transform, width, and height.
    """
    import rasterio
    from rasterio.warp import reproject

    with rasterio.open(str(reference_path)) as ref:
        profile = ref.profile.copy()
        profile.update({
            'count': src.count,
            'dtype': src.dtypes[0],
            'driver': 'GTiff',
            'compress': 'lzw',
        })

        with rasterio.open(str(output_path), 'w', **profile) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs or dst_crs,
                    dst_transform=ref.transform,
                    dst_crs=ref.crs,
                    resampling=resampling,
                )

    logger.info(f"Reprojected to match reference grid → {output_path}")
    return output_path


def reproject_array(
    array: np.ndarray,
    src_bbox: BoundingBox,
    dst_shape: tuple[int, int],
    dst_bbox: BoundingBox,
    method: str = "bilinear",
) -> np.ndarray:
    """Reproject a 2D array from one bounding box to another using interpolation.

    Pure NumPy/SciPy implementation — no GDAL dependency. Useful for quick
    reprojection of computed arrays (slope, aspect) that don't have CRS metadata.

    Args:
        array: Input 2D array.
        src_bbox: Source geographic bounding box.
        dst_shape: Desired output shape (rows, cols).
        dst_bbox: Destination geographic bounding box.
        method: Interpolation method — "nearest", "bilinear".

    Returns:
        Reprojected 2D array.
    """
    from scipy.interpolate import RegularGridInterpolator

    # Source coordinate grids
    src_lats = np.linspace(src_bbox.lat_max, src_bbox.lat_min, array.shape[0])
    src_lons = np.linspace(src_bbox.lon_min, src_bbox.lon_max, array.shape[1])

    # Destination coordinate grids
    dst_lats = np.linspace(dst_bbox.lat_max, dst_bbox.lat_min, dst_shape[0])
    dst_lons = np.linspace(dst_bbox.lon_min, dst_bbox.lon_max, dst_shape[1])
    dst_lon_grid, dst_lat_grid = np.meshgrid(dst_lons, dst_lats)

    # Interpolate
    method_map = {"nearest": "nearest", "bilinear": "linear"}
    interp = RegularGridInterpolator(
        (src_lats[::-1], src_lons),  # Must be ascending
        array[::-1],
        method=method_map.get(method, "linear"),
        bounds_error=False,
        fill_value=np.nan,
    )

    points = np.column_stack([
        dst_lat_grid.ravel(),
        dst_lon_grid.ravel(),
    ])
    result = interp(points).reshape(dst_shape)

    return result


def read_raster_as_array(path: str | Path) -> tuple[np.ndarray, dict]:
    """Read a raster file and return the data as a NumPy array with metadata.

    Args:
        path: Path to the raster file.

    Returns:
        Tuple of (array, metadata_dict). Array shape is (bands, rows, cols)
        or (rows, cols) for single-band rasters.
    """
    import rasterio

    path = Path(path)
    with rasterio.open(str(path)) as src:
        data = src.read()  # Shape: (bands, rows, cols)
        meta = {
            'crs': str(src.crs),
            'transform': list(src.transform),
            'bounds': list(src.bounds),
            'width': src.width,
            'height': src.height,
            'count': src.count,
            'dtype': str(src.dtypes[0]),
            'nodata': src.nodata,
        }

    # Squeeze single-band
    if data.shape[0] == 1:
        data = data[0]

    return data, meta
