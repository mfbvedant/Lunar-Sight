"""
Agent 2 — LangGraph Node Function
====================================
Complex-Valued SAR Despeckling agent.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import yaml
import numpy as np

from shared.state import LunarSightState
from shared.io_utils import load_tensor, save_tensor, get_output_dir

logger = logging.getLogger(__name__)


def agent2_node(state: LunarSightState) -> dict[str, Any]:
    """LangGraph node for Agent 2 — SAR Despeckling.

    If a trained model exists, runs inference. Otherwise, logs that
    training must be done via Colab notebook.

    Args:
        state: Current pipeline state.

    Returns:
        State updates dict.
    """
    logger.info("=" * 60)
    logger.info("AGENT 2 — Complex-Valued SAR Despeckling")
    logger.info("=" * 60)

    start_time = time.time()
    updates: dict[str, Any] = {
        "agent2_status": "running",
        "current_agent": "agent2",
    }

    try:
        config_path = state.get("mission_config_path", "config/mission_config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        output_dir = get_output_dir(config, "agent2")

        # Load input tensor from Agent 1
        raw_tensor_path = state.get("raw_tensor_path")
        if not raw_tensor_path or not Path(raw_tensor_path).exists():
            raise FileNotFoundError("Raw tensor from Agent 1 not found")

        raw_tensor = load_tensor(raw_tensor_path)
        logger.info(f"Input tensor: {raw_tensor.shape}")

        # Check for pre-trained model
        model_path = state.get("despeckling_model_path")
        checkpoint_dir = config.get("data", {}).get("checkpoint_dir", "./checkpoints")
        best_model = Path(checkpoint_dir) / "agent2" / "best_model.pt"

        if model_path and Path(model_path).exists():
            best_model = Path(model_path)

        if best_model.exists():
            logger.info(f"Found trained model: {best_model}")
            despeckled = _run_inference(raw_tensor, best_model, config)
        else:
            logger.warning(
                "No trained despeckling model found. "
                "Please run notebook 02_despeckling_training.ipynb on Colab. "
                "Passing through raw tensor as fallback."
            )
            despeckled = raw_tensor

        # Save output
        out_path = save_tensor(
            despeckled,
            output_dir / "despeckled_tensor.npy",
            metadata={"source": "agent2", "shape": list(despeckled.shape)},
        )

        elapsed = time.time() - start_time
        logger.info(f"Agent 2 completed in {elapsed:.1f}s")

        updates.update({
            "despeckled_tensor_path": str(out_path),
            "agent2_status": "success",
            "agent2_error": None,
        })

    except Exception as e:
        logger.error(f"Agent 2 failed: {e}", exc_info=True)
        updates.update({
            "agent2_status": "error",
            "agent2_error": str(e),
        })

    return updates


def _run_inference(tensor, model_path, config):
    """Load model and run inference."""
    import torch
    from agent2_despeckling.cv_cnn_model import create_model
    from agent2_despeckling.inference import run_inference, verify_phase_integrity

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(in_channels=tensor.shape[0], device=device)

    ckpt = torch.load(str(model_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    despeckled = run_inference(model, tensor, device=device)

    # Verify phase integrity
    stats = verify_phase_integrity(tensor, despeckled)
    logger.info(f"Phase integrity: {stats}")

    return despeckled
