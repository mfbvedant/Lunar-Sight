"""
Agent 4 — LangGraph Node Function
====================================
Weakly-supervised semantic segmentation agent.
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


def agent4_node(state: LunarSightState) -> dict[str, Any]:
    """LangGraph node for Agent 4 — Segmentation.

    If trained model exists → run inference to produce ice mask.
    Otherwise → generate pseudo-labels for Colab training.

    Args:
        state: Pipeline state.

    Returns:
        State updates.
    """
    logger.info("=" * 60)
    logger.info("AGENT 4 — Weakly-Supervised Semantic Segmentation")
    logger.info("=" * 60)

    start_time = time.time()
    updates: dict[str, Any] = {
        "agent4_status": "running",
        "current_agent": "agent4",
    }

    try:
        config_path = state.get("mission_config_path", "config/mission_config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        output_dir = get_output_dir(config, "agent4")

        # Load polarimetric tensor
        feat_path = state.get("polarimetric_tensor_path")
        if not feat_path or not Path(feat_path).exists():
            raise FileNotFoundError("Polarimetric feature tensor not found")

        feat_tensor = load_tensor(feat_path)
        logger.info(f"Feature tensor: {feat_tensor.shape}")

        # Check for trained model
        checkpoint_dir = config.get("data", {}).get("checkpoint_dir", "./checkpoints")
        best_model = Path(checkpoint_dir) / "agent4" / "best_model.pt"
        model_path = state.get("segmentation_model_path")
        if model_path and Path(model_path).exists():
            best_model = Path(model_path)

        if best_model.exists():
            logger.info(f"Found trained model: {best_model}")
            ice_mask, confidence = _run_inference(feat_tensor, best_model, config)
        else:
            logger.warning(
                "No trained segmentation model found. "
                "Generating pseudo-labels for Colab training."
            )
            ice_mask, confidence = _generate_pseudo_labels_and_pass(
                feat_tensor, config, output_dir
            )

        # Apply confidence threshold from state (may be lowered on retry)
        conf_threshold = state.get("agent4_confidence_threshold", 0.5)
        ice_mask_thresholded = (ice_mask == 1) & (confidence >= conf_threshold)
        ice_mask_final = ice_mask_thresholded.astype(np.uint8)

        # Save outputs
        mask_path = save_tensor(ice_mask_final, output_dir / "ice_mask.npy")
        conf_path = save_tensor(confidence, output_dir / "confidence_map.npy")

        logger.info(
            f"Ice pixels (threshold={conf_threshold:.2f}): {ice_mask_final.sum()} / {ice_mask_final.size}"
        )

        elapsed = time.time() - start_time
        logger.info(f"Agent 4 completed in {elapsed:.1f}s")

        updates.update({
            "ice_mask_path": str(mask_path),
            "confidence_map_path": str(conf_path),
            "agent4_status": "success",
            "agent4_error": None,
        })

    except Exception as e:
        logger.error(f"Agent 4 failed: {e}", exc_info=True)
        updates.update({
            "agent4_status": "error",
            "agent4_error": str(e),
        })

    return updates


def _run_inference(feat_tensor, model_path, config):
    """Load model and run inference."""
    import torch
    from agent4_segmentation.model import create_unet
    from agent4_segmentation.inference import run_segmentation_inference

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_unet(in_channels=feat_tensor.shape[0], device=device)

    ckpt = torch.load(str(model_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    return run_segmentation_inference(model, feat_tensor, device=device)


def _generate_pseudo_labels_and_pass(feat_tensor, config, output_dir):
    """Generate pseudo-labels and return them as the 'ice mask'."""
    from agent4_segmentation.pseudo_labels import generate_pseudo_labels

    thresholds = config.get("thresholds", {})

    # Extract CPR and DOP from feature tensor channels
    # Channel order depends on build_feature_tensor output
    l_cpr = feat_tensor[0]  # First channel is L_CPR
    dop = feat_tensor[2] if feat_tensor.shape[0] > 2 else feat_tensor[1]

    s_cpr = feat_tensor[1] if feat_tensor.shape[0] > 8 else None

    label_map, stats = generate_pseudo_labels(
        l_cpr=l_cpr,
        s_cpr=s_cpr,
        dop=dop,
        ice_cpr_min=thresholds.get("ice_cpr_min", 1.0),
        ice_dop_max=thresholds.get("ice_dop_max", 0.13),
    )

    save_tensor(label_map, output_dir / "pseudo_labels.npy")

    ice_mask = (label_map == 1.0).astype(np.uint8)
    confidence = np.where(np.isfinite(label_map), 1.0, 0.0).astype(np.float32)

    return ice_mask, confidence
