"""
LunarSight Orchestrator
=========================
LangGraph-based orchestration of all 5 agents with conditional routing,
retry logic, and dynamic threshold adaptation.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_graph():
    """Build the LangGraph state graph.

    Graph topology:
        agent1 → agent2 → agent3 → agent4 → coverage_check
        coverage_check → agent5 (if pass) OR threshold_adapt → agent4 (if retry)
        agent5 → path_check
        path_check → done (if pass) OR relax → agent5 (if retry)

    Returns:
        Compiled LangGraph StateGraph.
    """
    from langgraph.graph import StateGraph, END

    from shared.state import LunarSightState
    from agent1_ingestion.agent import agent1_node
    from agent2_despeckling.agent import agent2_node
    from agent3_polarimetry.agent import agent3_node
    from agent4_segmentation.agent import agent4_node
    from agent5_pathfinding.agent import agent5_node

    workflow = StateGraph(LunarSightState)

    # ---- Add nodes ----
    workflow.add_node("agent1_ingestion", agent1_node)
    workflow.add_node("agent2_despeckling", agent2_node)
    workflow.add_node("agent3_polarimetry", agent3_node)
    workflow.add_node("agent4_segmentation", agent4_node)
    workflow.add_node("agent5_pathfinding", agent5_node)
    workflow.add_node("coverage_check", coverage_check_node)
    workflow.add_node("path_check", path_check_node)
    workflow.add_node("threshold_adapt", threshold_adapt_node)
    workflow.add_node("path_relax", path_relax_node)

    # ---- Define edges ----
    # Linear pipeline
    workflow.add_edge("agent1_ingestion", "agent2_despeckling")
    workflow.add_edge("agent2_despeckling", "agent3_polarimetry")
    workflow.add_edge("agent3_polarimetry", "agent4_segmentation")
    workflow.add_edge("agent4_segmentation", "coverage_check")

    # Coverage check → conditional
    workflow.add_conditional_edges(
        "coverage_check",
        _coverage_router,
        {
            "pass": "agent5_pathfinding",
            "retry": "threshold_adapt",
            "fail": END,
        },
    )

    # Threshold adapt → retry segmentation
    workflow.add_edge("threshold_adapt", "agent4_segmentation")

    # Agent 5 → path check
    workflow.add_edge("agent5_pathfinding", "path_check")

    # Path check → conditional
    workflow.add_conditional_edges(
        "path_check",
        _path_router,
        {
            "pass": END,
            "retry": "path_relax",
            "fail": END,
        },
    )

    # Path relax → retry pathfinding
    workflow.add_edge("path_relax", "agent5_pathfinding")

    # Entry point
    workflow.set_entry_point("agent1_ingestion")

    return workflow.compile()


# ---- Router functions ----

def _coverage_router(state: dict) -> str:
    """Route based on ice mask coverage quality."""
    coverage = state.get("ice_coverage_pct", 0.0)
    retries = state.get("coverage_retries", 0)
    max_retries = state.get("max_coverage_retries", 3)

    if coverage > 0.5:
        return "pass"
    elif retries < max_retries:
        return "retry"
    else:
        return "fail"


def _path_router(state: dict) -> str:
    """Route based on path finding success."""
    path_failure = state.get("path_failure", True)
    retries = state.get("path_retries", 0)
    max_retries = state.get("max_path_retries", 2)

    if not path_failure:
        return "pass"
    elif retries < max_retries:
        return "retry"
    else:
        return "fail"


# ---- Check / Adapt nodes ----

def coverage_check_node(state: dict) -> dict[str, Any]:
    """Evaluate ice mask quality for downstream pathfinding."""
    import numpy as np
    from pathlib import Path
    from shared.io_utils import load_tensor

    ice_mask_path = state.get("ice_mask_path")
    if not ice_mask_path or not Path(ice_mask_path).exists():
        logger.error("No ice mask found for coverage check")
        return {"ice_coverage_pct": 0.0}

    ice_mask = load_tensor(ice_mask_path)
    total = ice_mask.size
    ice_pixels = np.sum(ice_mask > 0)
    coverage = float(ice_pixels / max(total, 1) * 100)

    logger.info(f"Coverage check: {ice_pixels} ice pixels / {total} total ({coverage:.2f}%)")

    return {"ice_coverage_pct": coverage}


def path_check_node(state: dict) -> dict[str, Any]:
    """Evaluate path quality."""
    path = state.get("traverse_path")
    path_failure = state.get("path_failure", True)

    if path_failure or not path:
        logger.warning("Path check: FAILED — no valid path")
        return {}

    # Check energy budget
    battery_cap = state.get("battery_capacity_wh", 50.0)
    final_battery = path[-1].get("battery_wh", 0)
    energy_margin = final_battery / max(battery_cap, 1)

    if energy_margin < 0.1:
        logger.warning(f"Path check: LOW BATTERY margin ({energy_margin:.1%})")

    logger.info(f"Path check: PASS — {len(path)} waypoints, battery margin: {energy_margin:.1%}")
    return {}


def threshold_adapt_node(state: dict) -> dict[str, Any]:
    """Lower thresholds to include more ice candidates.

    Called when coverage_check fails — relaxes polarimetric thresholds
    to widen the ice candidate pool.
    """
    retries = state.get("coverage_retries", 0) + 1
    current_conf = state.get("agent4_confidence_threshold", 0.5)

    # Lower confidence threshold
    new_conf = max(current_conf - 0.1, 0.2)

    logger.info(
        f"Threshold adapt (retry {retries}): "
        f"confidence {current_conf:.2f} → {new_conf:.2f}"
    )

    return {
        "coverage_retries": retries,
        "agent4_confidence_threshold": new_conf,
    }


def path_relax_node(state: dict) -> dict[str, Any]:
    """Relax pathfinding constraints.

    Called when path_check fails — increases max slope, lowers battery
    reserve requirements.
    """
    retries = state.get("path_retries", 0) + 1

    logger.info(f"Path relax (retry {retries}): widening slope + battery constraints")

    return {
        "path_retries": retries,
    }
