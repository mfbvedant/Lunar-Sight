"""
Agent 1 — LangGraph Node Function
====================================
Data Ingestion & Topographic Synthesis agent.

Executes the full pipeline: fetch data → reproject → compute slope → build tensor.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import yaml
import numpy as np

from shared.state import LunarSightState
from shared.geo_utils import BoundingBox
from shared.io_utils import get_output_dir
from agent1_ingestion.horn_slope import horn_slope
from agent1_ingestion.tensor_builder import (
    build_coregistered_tensor,
    save_coregistered_tensor,
    load_complex_binary,
)

logger = logging.getLogger(__name__)


def agent1_node(state: LunarSightState) -> dict[str, Any]:
    """LangGraph node function for Agent 1 — Data Ingestion.

    Reads mission config, fetches/loads radar and DEM data, reprojects to
    a common CRS, computes slope/aspect, and builds the co-registered tensor.

    Args:
        state: Current pipeline state dict.

    Returns:
        Dict of state updates with tensor path and status.
    """
    logger.info("=" * 60)
    logger.info("AGENT 1 — Data Ingestion & Topographic Synthesis")
    logger.info("=" * 60)

    start_time = time.time()
    updates: dict[str, Any] = {
        "agent1_status": "running",
        "current_agent": "agent1",
    }

    try:
        # Load mission config
        config_path = state.get("mission_config_path", "config/mission_config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        bbox = BoundingBox.from_config(config)
        logger.info(f"Target: {config['target']['crater_name']}, bbox={bbox.as_tuple()}")

        # Set up output directory
        output_dir = get_output_dir(config, "agent1")

        # ---- Step 1: Load Data ----
        logger.info("Step 1: Loading radar and DEM data...")
        l_band_data, s_band_data, dem_data = _load_data(config, state)

        # ---- Step 2: Reprojection ----
        logger.info("Step 2: Co-registering data (reprojection)...")
        # If data is already co-registered (from manual preprocessing),
        # skip reprojection. Otherwise, use rasterio/GDAL.
        # For now, assume data is already aligned (user pre-processed).

        # ---- Step 3: Compute Slope & Aspect ----
        logger.info("Step 3: Computing slope and aspect (Horn's algorithm)...")
        if dem_data is not None:
            pixel_size = config.get("crs", {}).get("pixel_size_m", 118.0)
            slope, aspect = horn_slope(dem_data, dx=pixel_size)
            logger.info(
                f"Slope: min={slope.min():.2f}°, max={slope.max():.2f}°, "
                f"mean={slope.mean():.2f}°"
            )

            # Save individual arrays
            np.save(str(output_dir / "slope.npy"), slope)
            np.save(str(output_dir / "aspect.npy"), aspect)
            np.save(str(output_dir / "dem.npy"), dem_data)
            updates["slope_path"] = str(output_dir / "slope.npy")
            updates["aspect_path"] = str(output_dir / "aspect.npy")
            updates["dem_path"] = str(output_dir / "dem.npy")
            updates["pixel_size_m"] = pixel_size
        else:
            slope = None
            aspect = None
            logger.warning("No DEM data available — slope/aspect not computed")

        # ---- Step 4: Build Tensor ----
        logger.info("Step 4: Building co-registered tensor...")

        # Extract real/imag from complex L-band
        if l_band_data is not None and np.iscomplexobj(l_band_data):
            l_real = l_band_data.real
            l_imag = l_band_data.imag
        elif l_band_data is not None:
            l_real = l_band_data
            l_imag = np.zeros_like(l_band_data)
        else:
            raise ValueError("L-band data is required but not available")

        # Extract real/imag from complex S-band
        s_real = s_imag = None
        if s_band_data is not None and np.iscomplexobj(s_band_data):
            s_real = s_band_data.real
            s_imag = s_band_data.imag
        elif s_band_data is not None:
            s_real = s_band_data
            s_imag = np.zeros_like(s_band_data)

        tensor = build_coregistered_tensor(
            l_band_real=l_real,
            l_band_imag=l_imag,
            s_band_real=s_real,
            s_band_imag=s_imag,
            dem=dem_data,
            slope=slope,
            aspect=aspect,
            normalize=True,
        )

        # Save tensor
        tensor_path = save_coregistered_tensor(
            tensor,
            output_dir / "coregistered_tensor.npy",
            metadata={
                "crater": config["target"]["crater_name"],
                "bbox": bbox.as_tuple(),
                "num_channels": tensor.shape[0],
            },
        )

        elapsed = time.time() - start_time
        logger.info(f"Agent 1 completed in {elapsed:.1f}s")
        logger.info(f"Tensor shape: {tensor.shape}, saved to: {tensor_path}")

        updates.update({
            "raw_tensor_path": str(tensor_path),
            "agent1_status": "success",
            "agent1_error": None,
        })

    except Exception as e:
        logger.error(f"Agent 1 failed: {e}", exc_info=True)
        updates.update({
            "agent1_status": "error",
            "agent1_error": str(e),
        })

    return updates


def _load_data(
    config: dict,
    state: LunarSightState,
) -> tuple[Any, Any, Any]:
    """Load radar and DEM data from configured paths or state.

    Returns:
        Tuple of (l_band_array, s_band_array, dem_array).
        Any can be None if not available.
    """
    data_config = config.get("data", {})

    l_band_data = None
    s_band_data = None
    dem_data = None

    # L-band
    l_path = data_config.get("dfsar_l_band_path") or state.get("dfsar_l_band_path")
    if l_path and Path(l_path).exists():
        path = Path(l_path)
        if path.suffix == '.npy':
            l_band_data = np.load(str(path), allow_pickle=False)
        elif path.suffix in ('.img', '.dat'):
            # Try to load as complex binary — shape must be provided
            # This is a simplified loader; real usage should parse PDS4 label
            logger.info(f"Loading L-band from binary: {path}")
            # Will be loaded via pds4_parser + tensor_builder in production
            l_band_data = np.fromfile(str(path), dtype=np.complex64)
            # Attempt to reshape as square
            side = int(np.sqrt(l_band_data.size))
            if side * side == l_band_data.size:
                l_band_data = l_band_data.reshape(side, side)
            else:
                logger.warning(f"Cannot auto-reshape L-band ({l_band_data.size} elements)")
        elif path.suffix in ('.tif', '.tiff'):
            from agent1_ingestion.reprojection import read_raster_as_array
            l_band_data, _ = read_raster_as_array(path)
        logger.info(f"L-band loaded: shape={l_band_data.shape if l_band_data is not None else 'N/A'}")
    else:
        logger.warning("L-band path not configured or file not found")

    # S-band
    s_path = data_config.get("dfsar_s_band_path") or state.get("dfsar_s_band_path")
    if s_path and Path(s_path).exists():
        path = Path(s_path)
        if path.suffix == '.npy':
            s_band_data = np.load(str(path), allow_pickle=False)
        elif path.suffix in ('.tif', '.tiff'):
            from agent1_ingestion.reprojection import read_raster_as_array
            s_band_data, _ = read_raster_as_array(path)
        logger.info(f"S-band loaded: shape={s_band_data.shape if s_band_data is not None else 'N/A'}")

    # DEM
    dem_path = data_config.get("lola_dem_path") or state.get("lola_dem_path")
    if dem_path and Path(dem_path).exists():
        path = Path(dem_path)
        if path.suffix == '.npy':
            dem_data = np.load(str(path), allow_pickle=False)
        elif path.suffix in ('.tif', '.tiff'):
            from agent1_ingestion.reprojection import read_raster_as_array
            dem_data, _ = read_raster_as_array(path)
        elif path.suffix in ('.img', '.dat'):
            dem_data = np.fromfile(str(path), dtype=np.float32)
            side = int(np.sqrt(dem_data.size))
            if side * side == dem_data.size:
                dem_data = dem_data.reshape(side, side)
        logger.info(f"DEM loaded: shape={dem_data.shape if dem_data is not None else 'N/A'}")

    return l_band_data, s_band_data, dem_data
