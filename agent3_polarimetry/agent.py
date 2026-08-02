"""
Agent 3 — LangGraph Node Function
====================================
Polarimetric Feature Extraction agent.

Takes despeckled covariance data → computes Stokes, CPR, m-χ, thresholds →
outputs a Polarimetric Feature Tensor.
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
from agent3_polarimetry.stokes import (
    compute_stokes_from_covariance,
    validate_stokes,
)
from agent3_polarimetry.cpr import compute_cpr, cpr_statistics
from agent3_polarimetry.mchi import (
    compute_mchi,
    compute_degree_of_polarization,
    dominant_scattering_mechanism,
)
from agent3_polarimetry.thresholds import (
    ThresholdConfig,
    generate_diagnostic_flags,
)
from agent3_polarimetry.feature_tensor import (
    build_feature_tensor,
    save_feature_tensor,
)

logger = logging.getLogger(__name__)


def agent3_node(state: LunarSightState) -> dict[str, Any]:
    """LangGraph node function for Agent 3 — Polarimetric Feature Extraction.

    Pipeline:
        1. Load despeckled covariance tensor from Agent 2
        2. Compute Stokes parameters (S₁-S₄)
        3. Compute CPR (L-band and S-band)
        4. Compute m-χ decomposition (m, R, G, B)
        5. Apply threshold-based diagnostic flags
        6. Build and save Polarimetric Feature Tensor

    Args:
        state: Current pipeline state dict.

    Returns:
        Dict of state updates.
    """
    logger.info("=" * 60)
    logger.info("AGENT 3 — Polarimetric Feature Extraction")
    logger.info("=" * 60)

    start_time = time.time()
    updates: dict[str, Any] = {
        "agent3_status": "running",
        "current_agent": "agent3",
    }

    try:
        # Load config
        config_path = state.get("mission_config_path", "config/mission_config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        output_dir = get_output_dir(config, "agent3")
        threshold_config = ThresholdConfig.from_config(config)

        # ---- Load despeckled data ----
        logger.info("Loading despeckled covariance data...")
        despeckled_path = state.get("despeckled_tensor_path")
        if not despeckled_path or not Path(despeckled_path).exists():
            # Fallback: use raw tensor if despeckling was skipped
            logger.warning(
                "Despeckled tensor not found, using raw tensor from Agent 1"
            )
            despeckled_path = state.get("raw_tensor_path")

        if not despeckled_path or not Path(despeckled_path).exists():
            raise FileNotFoundError(
                "No input tensor available. Run Agent 1 (and optionally Agent 2) first."
            )

        tensor = load_tensor(despeckled_path)
        logger.info(f"Loaded tensor: shape={tensor.shape}")

        # ---- Extract covariance elements ----
        # Channel layout from Agent 1/2:
        # 0: L-band real, 1: L-band imag, 2: S-band real, 3: S-band imag, ...
        c11, c12, c21, c22 = _extract_covariance(tensor)

        # ---- Step 1: Stokes Parameters ----
        logger.info("Computing Stokes parameters...")
        s1, s2, s3, s4 = compute_stokes_from_covariance(c11, c12, c21, c22)
        stokes_stats = validate_stokes(s1, s2, s3, s4)
        logger.info(f"Stokes validation: {stokes_stats}")

        # Save Stokes
        stokes_arr = np.stack([s1, s2, s3, s4], axis=0)
        stokes_path = output_dir / "stokes_parameters.npy"
        np.save(str(stokes_path), stokes_arr)
        updates["stokes_path"] = str(stokes_path)

        # ---- Step 2: CPR ----
        logger.info("Computing CPR...")
        l_cpr = compute_cpr(s1, s4)
        cpr_stats = cpr_statistics(l_cpr)
        logger.info(f"L-band CPR stats: {cpr_stats}")

        np.save(str(output_dir / "l_cpr.npy"), l_cpr)
        updates["cpr_l_path"] = str(output_dir / "l_cpr.npy")

        # S-band CPR (if available)
        s_cpr = None
        if tensor.shape[0] >= 4:
            # Compute S-band Stokes from channels 2-3
            s_c11, s_c12, s_c21, s_c22 = _extract_covariance_sband(tensor)
            s_s1, _, _, s_s4 = compute_stokes_from_covariance(
                s_c11, s_c12, s_c21, s_c22
            )
            s_cpr = compute_cpr(s_s1, s_s4)
            np.save(str(output_dir / "s_cpr.npy"), s_cpr)
            updates["cpr_s_path"] = str(output_dir / "s_cpr.npy")
            logger.info(f"S-band CPR stats: {cpr_statistics(s_cpr)}")

        # ---- Step 3: m-χ Decomposition ----
        logger.info("Computing m-χ decomposition...")
        mchi = compute_mchi(s1, s2, s3, s4)
        logger.info(f"DOP (m): mean={np.nanmean(mchi['m']):.4f}")

        dop = mchi['m']
        dom_mech = dominant_scattering_mechanism(mchi['R'], mchi['G'], mchi['B'])

        # ---- Step 4: Threshold Classification ----
        logger.info("Applying diagnostic thresholds...")
        flags = generate_diagnostic_flags(
            l_cpr=l_cpr,
            dop=dop,
            m=mchi['m'],
            dominant_mechanism=dom_mech,
            s_cpr=s_cpr,
            thresholds=threshold_config,
        )

        # ---- Step 5: Build Feature Tensor ----
        logger.info("Building Polarimetric Feature Tensor...")
        feat_tensor, channel_names = build_feature_tensor(
            l_cpr=l_cpr,
            dop=dop,
            mchi_R=mchi['R'],
            mchi_G=mchi['G'],
            mchi_B=mchi['B'],
            ice_flag=flags['ice_flag'],
            rock_flag=flags['rock_flag'],
            s_cpr=s_cpr,
            s1=s1,
        )

        feat_path = save_feature_tensor(
            feat_tensor,
            channel_names,
            output_dir / "polarimetric_feature_tensor.npy",
            metadata={
                "crater": config["target"]["crater_name"],
                "threshold_config": {
                    "ice_cpr_min": threshold_config.ice_cpr_min,
                    "ice_dop_max": threshold_config.ice_dop_max,
                },
            },
        )

        elapsed = time.time() - start_time
        logger.info(f"Agent 3 completed in {elapsed:.1f}s")
        logger.info(f"Feature tensor: {feat_tensor.shape}, channels={channel_names}")

        updates.update({
            "polarimetric_tensor_path": str(feat_path),
            "agent3_status": "success",
            "agent3_error": None,
        })

    except Exception as e:
        logger.error(f"Agent 3 failed: {e}", exc_info=True)
        updates.update({
            "agent3_status": "error",
            "agent3_error": str(e),
        })

    return updates


def _extract_covariance(tensor: np.ndarray):
    """Extract C₂ covariance matrix elements from L-band channels.

    From the co-registered tensor (channels 0,1 = L real/imag):
        E_H = channel_0 + j * channel_1
        C₁₁ = |E_H|², C₁₂ = E_H * conj(E_H) (simplified for single-pol)

    For full dual-pol, this should use both HH and HV channels.
    """
    if tensor.ndim == 3:
        l_real = tensor[0]
        l_imag = tensor[1]
    elif tensor.ndim == 2:
        l_real = tensor
        l_imag = np.zeros_like(tensor)
    else:
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")

    e_h = l_real + 1j * l_imag

    # Simplified C₂ for single complex channel
    # In production, this should use HH/HV channels separately
    c11 = np.abs(e_h) ** 2
    c12 = e_h * np.conj(e_h)  # This is just |E_H|² for single-channel
    c21 = np.conj(c12)
    c22 = c11 * 0.5  # Placeholder — should come from VV channel

    return c11, c12, c21, c22


def _extract_covariance_sband(tensor: np.ndarray):
    """Extract C₂ elements from S-band channels (2, 3)."""
    if tensor.shape[0] < 4:
        raise ValueError("Tensor has fewer than 4 channels — no S-band data")

    s_real = tensor[2]
    s_imag = tensor[3]
    e_h = s_real + 1j * s_imag

    c11 = np.abs(e_h) ** 2
    c12 = e_h * np.conj(e_h)
    c21 = np.conj(c12)
    c22 = c11 * 0.5

    return c11, c12, c21, c22
