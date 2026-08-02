"""
Polarimetric Feature Tensor Builder
=====================================
Stack all computed polarimetric layers into a single multi-channel feature
tensor for downstream segmentation (Agent 4).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Any

import numpy as np

from shared.io_utils import save_tensor

logger = logging.getLogger(__name__)

# Default channel ordering
DEFAULT_CHANNELS = [
    "L_CPR",
    "S_CPR",
    "DOP",          # degree of polarization (m)
    "R_dbl",        # m-χ red (double-bounce)
    "G_vol",        # m-χ green (volume)
    "B_srf",        # m-χ blue (surface)
    "ice_flag",     # binary ice candidate
    "rock_flag",    # binary rock candidate
    "S1_total",     # total power (L-band)
]


def build_feature_tensor(
    l_cpr: np.ndarray,
    dop: np.ndarray,
    mchi_R: np.ndarray,
    mchi_G: np.ndarray,
    mchi_B: np.ndarray,
    ice_flag: np.ndarray,
    rock_flag: np.ndarray,
    s_cpr: Optional[np.ndarray] = None,
    s1: Optional[np.ndarray] = None,
    extra_channels: Optional[dict[str, np.ndarray]] = None,
) -> tuple[np.ndarray, list[str]]:
    """Build the Polarimetric Feature Tensor from individual layers.

    Args:
        l_cpr: L-band CPR (H×W).
        dop: Degree of polarization (H×W).
        mchi_R: m-χ red / double-bounce (H×W).
        mchi_G: m-χ green / volume (H×W).
        mchi_B: m-χ blue / surface (H×W).
        ice_flag: Binary ice candidate flag (H×W).
        rock_flag: Binary rock candidate flag (H×W).
        s_cpr: S-band CPR (H×W), optional.
        s1: Total power S₁ (H×W), optional.
        extra_channels: Dict of additional named channels, optional.

    Returns:
        Tuple of:
            - Feature tensor (C, H, W), dtype=float32.
            - List of channel names in order.
    """
    reference_shape = l_cpr.shape
    channels: list[np.ndarray] = []
    names: list[str] = []

    def _add(arr: np.ndarray, name: str):
        if arr.shape != reference_shape:
            logger.warning(
                f"Channel '{name}': shape {arr.shape} != {reference_shape}. Resizing."
            )
            from scipy.ndimage import zoom
            factors = (
                reference_shape[0] / arr.shape[0],
                reference_shape[1] / arr.shape[1],
            )
            arr = zoom(arr.astype(np.float64), factors, order=1)

        # Convert bool to float
        if arr.dtype == bool:
            arr = arr.astype(np.float32)

        channels.append(np.nan_to_num(arr, nan=0.0).astype(np.float32))
        names.append(name)

    # Core channels
    _add(l_cpr, "L_CPR")

    if s_cpr is not None:
        _add(s_cpr, "S_CPR")

    _add(dop, "DOP")
    _add(mchi_R, "R_dbl")
    _add(mchi_G, "G_vol")
    _add(mchi_B, "B_srf")
    _add(ice_flag, "ice_flag")
    _add(rock_flag, "rock_flag")

    if s1 is not None:
        _add(s1, "S1_total")

    # Extra user-defined channels
    if extra_channels:
        for ch_name, ch_arr in extra_channels.items():
            _add(ch_arr, ch_name)

    # Stack: (C, H, W)
    tensor = np.stack(channels, axis=0)

    logger.info(
        f"Built feature tensor: shape={tensor.shape}, "
        f"channels={names}"
    )

    return tensor, names


def save_feature_tensor(
    tensor: np.ndarray,
    channel_names: list[str],
    output_path: str | Path,
    metadata: Optional[dict[str, Any]] = None,
) -> Path:
    """Save the feature tensor with channel metadata.

    Args:
        tensor: Feature tensor (C, H, W).
        channel_names: Ordered list of channel names.
        output_path: Output file path.
        metadata: Additional metadata.

    Returns:
        Path to saved tensor file.
    """
    meta = metadata or {}
    meta.update({
        'num_channels': tensor.shape[0],
        'height': tensor.shape[1],
        'width': tensor.shape[2],
        'channel_names': channel_names,
        'dtype': str(tensor.dtype),
    })

    return save_tensor(tensor, output_path, metadata=meta)
