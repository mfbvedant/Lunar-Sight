"""
LunarSight — Main Entry Point
================================
CLI entry point for running the full pipeline or individual agents.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml


def setup_logging(level: str = "INFO"):
    """Configure logging with colored output."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="LunarSight — Multi-Agent Lunar Ice Detection & Pathfinding"
    )
    parser.add_argument(
        "--config", "-c",
        default="config/mission_config.yaml",
        help="Path to mission config YAML",
    )
    parser.add_argument(
        "--agent",
        choices=["1", "2", "3", "4", "5", "all"],
        default="all",
        help="Run a specific agent or the full pipeline",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    logger = logging.getLogger("lunarsight.main")

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info("=" * 60)
    logger.info("LunarSight Pipeline")
    logger.info(f"Target: {config.get('target', {}).get('crater_name', 'Unknown')}")
    logger.info(f"Agent: {args.agent}")
    logger.info("=" * 60)

    if args.agent == "all":
        _run_full_pipeline(config, config_path)
    else:
        _run_single_agent(int(args.agent), config, config_path)


def _run_full_pipeline(config, config_path):
    """Run the complete LangGraph pipeline."""
    logger = logging.getLogger("lunarsight.main")

    try:
        from orchestrator import build_graph
        graph = build_graph()

        initial_state = {
            "mission_config_path": str(config_path),
            "current_agent": "agent1",
        }

        logger.info("Starting full pipeline...")
        result = graph.invoke(initial_state)

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info(f"Final status: Agent 5 = {result.get('agent5_status', 'unknown')}")
        if result.get("traverse_path"):
            logger.info(f"Path: {len(result['traverse_path'])} waypoints")
            logger.info(f"Distance: {result.get('path_distance_m', 0):.1f}m")
        logger.info("=" * 60)

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.info("Install with: pip install langgraph")
        sys.exit(1)


def _run_single_agent(agent_num, config, config_path):
    """Run a single agent with a stub state."""
    logger = logging.getLogger("lunarsight.main")

    state = {"mission_config_path": str(config_path)}

    agent_map = {
        1: ("agent1_ingestion.agent", "agent1_node"),
        2: ("agent2_despeckling.agent", "agent2_node"),
        3: ("agent3_polarimetry.agent", "agent3_node"),
        4: ("agent4_segmentation.agent", "agent4_node"),
        5: ("agent5_pathfinding.agent", "agent5_node"),
    }

    module_name, func_name = agent_map[agent_num]

    try:
        import importlib
        mod = importlib.import_module(module_name)
        node_fn = getattr(mod, func_name)

        logger.info(f"Running Agent {agent_num} standalone...")
        result = node_fn(state)

        status = result.get(f"agent{agent_num}_status", "unknown")
        error = result.get(f"agent{agent_num}_error")

        logger.info(f"Agent {agent_num} finished: {status}")
        if error:
            logger.error(f"Error: {error}")

    except ImportError as e:
        logger.error(f"Cannot import Agent {agent_num}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
