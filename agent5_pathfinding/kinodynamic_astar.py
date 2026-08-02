"""
Kinodynamic A* Pathfinder
===========================
4D search over (x, y, heading, time) with battery and thermal constraints,
slip-aware edge costs, and return-trip awareness.
"""

from __future__ import annotations

import heapq
import logging
import math
import time as time_module
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from agent5_pathfinding.cost_function import compute_edge_cost
from agent5_pathfinding.heuristic import combined_heuristic
from agent5_pathfinding.illumination import compute_temperature

logger = logging.getLogger(__name__)


@dataclass(order=True)
class SearchNode:
    """A* search node in the 4D state space."""

    priority: float
    row: int = field(compare=False)
    col: int = field(compare=False)
    heading: float = field(compare=False)      # degrees
    time_s: float = field(compare=False)        # seconds from start
    battery_wh: float = field(compare=False)    # remaining battery
    temperature_k: float = field(compare=False) # current temp
    g_cost: float = field(compare=False)        # cost from start
    parent: Optional[tuple] = field(compare=False, default=None)


# 8-connected neighborhood (row_offset, col_offset)
NEIGHBORS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def kinodynamic_astar(
    slope: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    pixel_size_m: float = 118.0,
    max_slope_deg: float = 10.0,
    mass_kg: float = 27.0,
    battery_capacity_wh: float = 50.0,
    power_nominal_w: float = 15.0,
    velocity_ms: float = 0.01,
    thermal_min_k: float = 173.0,
    start_temp_k: float = 250.0,
    max_iterations: int = 500_000,
    illumination_map: Optional[np.ndarray] = None,
) -> Optional[list[dict]]:
    """Run kinodynamic A* pathfinding from start to goal.

    Search state: (row, col, heading, time, battery, temperature).
    Constraints: battery > 0, temperature > min, slope < max.

    Args:
        slope: 2D slope array (degrees).
        start: (row, col) start position.
        goal: (row, col) goal position.
        pixel_size_m: Ground resolution.
        max_slope_deg: Max traversable slope.
        mass_kg: Rover mass.
        battery_capacity_wh: Full battery capacity.
        power_nominal_w: Nominal power draw.
        velocity_ms: Nominal velocity.
        thermal_min_k: Minimum operating temp.
        start_temp_k: Initial temperature.
        max_iterations: Max search iterations.
        illumination_map: Optional binary illumination (1=lit).

    Returns:
        List of waypoint dicts, or None if no path found.
    """
    h, w = slope.shape
    logger.info(
        f"A* search: ({start[0]},{start[1]}) → ({goal[0]},{goal[1]}), "
        f"grid={h}×{w}"
    )

    search_start = time_module.time()

    # Priority queue
    start_node = SearchNode(
        priority=0.0,
        row=start[0], col=start[1],
        heading=0.0, time_s=0.0,
        battery_wh=battery_capacity_wh,
        temperature_k=start_temp_k,
        g_cost=0.0,
        parent=None,
    )

    open_set = [start_node]
    closed_set = set()
    came_from: dict[tuple[int, int], SearchNode] = {}
    g_scores: dict[tuple[int, int], float] = {start: 0.0}

    iterations = 0
    nodes_explored = 0

    while open_set and iterations < max_iterations:
        iterations += 1
        current = heapq.heappop(open_set)

        pos = (current.row, current.col)

        if pos == goal:
            elapsed = time_module.time() - search_start
            logger.info(
                f"Path found! Iterations: {iterations}, "
                f"Explored: {nodes_explored}, Time: {elapsed:.2f}s"
            )
            return _reconstruct_path(current, came_from, start)

        if pos in closed_set:
            continue

        closed_set.add(pos)
        nodes_explored += 1

        # Expand neighbors
        for dr, dc in NEIGHBORS_8:
            nr, nc = current.row + dr, current.col + dc

            # Bounds check
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                continue

            npos = (nr, nc)
            if npos in closed_set:
                continue

            # Get slope at neighbor
            neighbor_slope = slope[nr, nc]
            if np.isnan(neighbor_slope):
                continue

            # Distance (diagonal vs cardinal)
            dist = pixel_size_m * (math.sqrt(2) if (dr != 0 and dc != 0) else 1.0)

            # Edge cost
            cost, energy, time_s = compute_edge_cost(
                neighbor_slope, dist,
                max_slope_deg=max_slope_deg,
                mass_kg=mass_kg,
                power_nominal_w=power_nominal_w,
                velocity_ms=velocity_ms,
            )

            if math.isinf(cost):
                continue

            # Update state
            new_battery = current.battery_wh - energy
            new_time = current.time_s + time_s
            new_heading = math.degrees(math.atan2(dc, -dr)) % 360

            # Illumination-based temperature
            illuminated = True
            if illumination_map is not None:
                illuminated = bool(illumination_map[nr, nc])

            time_in_shadow = 0.0 if illuminated else time_s / 3600.0
            new_temp = compute_temperature(
                illuminated, time_in_shadow, current.temperature_k
            )

            # Constraint checks
            if new_battery <= 0:
                continue
            if new_temp < thermal_min_k:
                continue

            new_g = current.g_cost + cost

            if npos in g_scores and new_g >= g_scores[npos]:
                continue

            g_scores[npos] = new_g

            h_val = combined_heuristic(
                npos, goal,
                battery_wh=new_battery,
                temp_k=new_temp,
                pixel_size_m=pixel_size_m,
                start=start,
            )

            neighbor_node = SearchNode(
                priority=new_g + h_val,
                row=nr, col=nc,
                heading=new_heading,
                time_s=new_time,
                battery_wh=new_battery,
                temperature_k=new_temp,
                g_cost=new_g,
                parent=pos,
            )

            came_from[npos] = current
            heapq.heappush(open_set, neighbor_node)

    elapsed = time_module.time() - search_start
    logger.warning(
        f"A* failed to find path. Iterations: {iterations}, "
        f"Explored: {nodes_explored}, Time: {elapsed:.2f}s"
    )
    return None


def _reconstruct_path(
    goal_node: SearchNode,
    came_from: dict,
    start: tuple[int, int],
) -> list[dict]:
    """Reconstruct the path from goal back to start."""
    path = []
    current = goal_node

    # Build from goal to start
    waypoint = {
        'x': current.col, 'y': current.row,
        'heading': current.heading,
        'time_s': current.time_s,
        'battery_wh': current.battery_wh,
        'temperature_k': current.temperature_k,
        'slip': 0.0,
        'cost': current.g_cost,
    }
    path.append(waypoint)

    pos = (current.row, current.col)
    while pos in came_from and pos != start:
        prev = came_from[pos]
        waypoint = {
            'x': prev.col, 'y': prev.row,
            'heading': prev.heading,
            'time_s': prev.time_s,
            'battery_wh': prev.battery_wh,
            'temperature_k': prev.temperature_k,
            'slip': 0.0,
            'cost': prev.g_cost,
        }
        path.append(waypoint)
        pos = (prev.row, prev.col)

    path.reverse()

    logger.info(
        f"Path: {len(path)} waypoints, "
        f"total cost: {path[-1]['cost']:.2f}, "
        f"battery used: {path[0].get('battery_wh', 0) - path[-1]['battery_wh']:.2f} Wh"
    )

    return path
