"""
Agent 5 — LangGraph Node Function
====================================
Terramechanic-Aware Spatiotemporal Pathfinding agent.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import yaml
import numpy as np

from shared.state import LunarSightState
from shared.io_utils import load_tensor, get_output_dir
from agent5_pathfinding.rover_config import RoverConfig
from agent5_pathfinding.graph_builder import (
    build_traversability_mask,
    find_best_ice_target,
)
from agent5_pathfinding.kinodynamic_astar import kinodynamic_astar

logger = logging.getLogger(__name__)


def agent5_node(state: LunarSightState) -> dict[str, Any]:
    """LangGraph node for Agent 5 — Pathfinding.

    Takes ice mask + DEM + slope → plans physically survivable traverse.

    Args:
        state: Pipeline state.

    Returns:
        State updates with traverse path or failure flag.
    """
    logger.info("=" * 60)
    logger.info("AGENT 5 — Terramechanic-Aware Pathfinding")
    logger.info("=" * 60)

    start_time = time.time()
    updates: dict[str, Any] = {
        "agent5_status": "running",
        "current_agent": "agent5",
        "path_failure": False,
    }

    try:
        config_path = state.get("mission_config_path", "config/mission_config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        output_dir = get_output_dir(config, "agent5")
        rover = RoverConfig.from_config(config)

        # Load inputs
        ice_mask_path = state.get("ice_mask_path")
        slope_path = state.get("slope_path")
        dem_path = state.get("dem_path")

        if not all(p and Path(p).exists() for p in [ice_mask_path, slope_path]):
            raise FileNotFoundError("Ice mask or slope data not found")

        ice_mask = load_tensor(ice_mask_path)
        slope = load_tensor(slope_path)
        pixel_size = state.get("pixel_size_m", 118.0)

        # Load confidence map
        conf_path = state.get("confidence_map_path")
        if conf_path and Path(conf_path).exists():
            confidence = load_tensor(conf_path)
        else:
            confidence = np.where(ice_mask > 0, 1.0, 0.0)

        # Build traversability mask
        traversable = build_traversability_mask(
            slope, rover.max_slope_deg, rover.mass_kg
        )

        # Select start (rim) and target (ice deposit)
        # Default: top-center of the grid as rim entry point
        start = (0, slope.shape[1] // 2)

        target = find_best_ice_target(
            ice_mask, confidence, traversable, start
        )

        if target is None:
            logger.error("No reachable ice deposit found")
            updates.update({
                "path_failure": True,
                "agent5_status": "error",
                "agent5_error": "No reachable ice deposit",
            })
            return updates

        # Run pathfinding
        logger.info(f"Planning path: {start} → {target}")
        path = kinodynamic_astar(
            slope=slope,
            start=start,
            goal=target,
            pixel_size_m=pixel_size,
            max_slope_deg=rover.max_slope_deg,
            mass_kg=rover.mass_kg,
            battery_capacity_wh=rover.battery_capacity_wh,
            power_nominal_w=rover.power_draw_nominal_w,
            velocity_ms=rover.max_velocity_ms,
            thermal_min_k=rover.thermal_min_temp_k,
        )

        if path is None:
            logger.error("A* failed to find a path")
            updates.update({
                "path_failure": True,
                "agent5_status": "error",
                "agent5_error": "No valid path found by A*",
            })
            return updates

        # Compute path statistics
        total_distance = sum(
            np.sqrt(
                (path[i+1]['y'] - path[i]['y']) ** 2 +
                (path[i+1]['x'] - path[i]['x']) ** 2
            ) * pixel_size
            for i in range(len(path) - 1)
        )
        energy_used = path[0].get('battery_wh', rover.battery_capacity_wh) - path[-1]['battery_wh']

        elapsed = time.time() - start_time
        logger.info(
            f"Path found: {len(path)} waypoints, "
            f"distance={total_distance:.1f}m, "
            f"energy={energy_used:.2f}Wh, "
            f"time={elapsed:.1f}s"
        )

        # Save path
        import json
        path_file = output_dir / "traverse_path.json"
        with open(path_file, 'w') as f:
            json.dump(path, f, indent=2)

        updates.update({
            "traverse_path": path,
            "path_distance_m": float(total_distance),
            "path_energy_wh": float(energy_used),
            "path_max_slip": 0.0,
            "path_failure": False,
            "agent5_status": "success",
            "agent5_error": None,
        })

    except Exception as e:
        logger.error(f"Agent 5 failed: {e}", exc_info=True)
        updates.update({
            "agent5_status": "error",
            "agent5_error": str(e),
            "path_failure": True,
        })

    return updates
