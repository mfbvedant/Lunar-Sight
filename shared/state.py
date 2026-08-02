"""
LunarSight State Schema
========================
Central TypedDict state shared across all agents via LangGraph StateGraph.
Every agent reads its inputs from this state and writes its outputs back.
"""

from __future__ import annotations

from typing import TypedDict, Optional


class LunarSightState(TypedDict, total=False):
    """Global pipeline state flowing through the LangGraph StateGraph.

    Fields marked with ``total=False`` are optional — agents populate them
    progressively as the pipeline executes.
    """

    # ---- Mission Configuration ----
    target_bbox: tuple[float, float, float, float]  # (lat_min, lat_max, lon_min, lon_max)
    crater_name: str
    mission_config_path: str
    training_config_path: str
    rover_config_path: str

    # ---- Agent 1 — Data Ingestion Outputs ----
    raw_tensor_path: str         # Path to co-registered multi-channel tensor (.npy / .h5)
    dem_path: str                # Path to reprojected DEM raster
    slope_path: str              # Path to Horn's slope array
    aspect_path: str             # Path to Horn's aspect array
    pixel_size_m: float          # Ground resolution in meters
    agent1_status: str           # "pending" | "running" | "success" | "error"
    agent1_error: Optional[str]

    # ---- Agent 2 — Despeckling Outputs ----
    despeckled_tensor_path: str  # Path to despeckled covariance tensor
    despeckling_model_path: str  # Path to trained CV-CNN weights
    agent2_status: str
    agent2_error: Optional[str]

    # ---- Agent 3 — Polarimetry Outputs ----
    polarimetric_tensor_path: str  # Path to Polarimetric Feature Tensor
    stokes_path: str               # Path to Stokes parameter arrays
    cpr_l_path: str                # Path to L-band CPR
    cpr_s_path: str                # Path to S-band CPR
    agent3_status: str
    agent3_error: Optional[str]

    # ---- Agent 4 — Segmentation Outputs ----
    ice_mask_path: str             # Path to binary ice mask (GeoTIFF)
    confidence_map_path: str       # Path to confidence map (GeoTIFF)
    segmentation_model_path: str   # Path to trained U-Net weights
    agent4_status: str
    agent4_error: Optional[str]
    agent4_confidence_threshold: float  # Adjustable by supervisor on retry

    # ---- Agent 5 — Pathfinding Outputs ----
    traverse_path: list[dict]      # List of waypoint dicts: {x, y, θ, t, battery, slip}
    path_energy_wh: float          # Total energy consumed by traverse
    path_distance_m: float         # Total path distance in meters
    path_max_slip: float           # Maximum slip ratio encountered
    path_failure: bool             # True if A* could not find a valid path
    agent5_status: str
    agent5_error: Optional[str]

    # ---- Orchestrator / System ----
    retry_count: int               # Current retry iteration count
    max_retries: int               # Maximum allowed retries (from config)
    current_agent: str             # Name of agent currently executing
    error_log: list[str]           # Accumulated error messages
    pipeline_status: str           # "running" | "success" | "failed"
