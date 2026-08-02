"""
Tensor Builder
===============
Build co-registered multi-channel tensors from radar data, DEM, and
derived topographic products (slope, aspect).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Any

import numpy as np

from shared.io_utils import save_tensor, save_tensor_h5

logger = logging.getLogger(__name__)


def build_coregistered_tensor(
    l_band_real: np.ndarray,
    l_band_imag: np.ndarray,
    s_band_real: Optional[np.ndarray] = None,
    s_band_imag: Optional[np.ndarray] = None,
    dem: Optional[np.ndarray] = None,
    slope: Optional[np.ndarray] = None,
    aspect: Optional[np.ndarray] = None,
    normalize: bool = True,
) -> np.ndarray:
    """Stack all data layers into a single multi-channel tensor.

    Channel layout (when all inputs provided):
        - 0: L-band real part
        - 1: L-band imaginary part
        - 2: S-band real part
        - 3: S-band imaginary part
        - 4: LOLA DEM elevation
        - 5: Horn's slope (degrees)
        - 6: Horn's aspect (degrees)

    If S-band is not available, channels 2-3 are omitted and subsequent
    channels shift down accordingly.

    Args:
        l_band_real: L-band real component (H×W).
        l_band_imag: L-band imaginary component (H×W).
        s_band_real: S-band real component (H×W), optional.
        s_band_imag: S-band imaginary component (H×W), optional.
        dem: DEM elevation array (H×W), optional.
        slope: Slope array in degrees (H×W), optional.
        aspect: Aspect array in degrees (H×W), optional.
        normalize: If True, normalize each channel to zero-mean, unit-variance.

    Returns:
        Multi-channel tensor of shape (C, H, W) where C = number of channels.
    """
    reference_shape = l_band_real.shape
    channels: list[np.ndarray] = []
    channel_names: list[str] = []

    # L-band (always present)
    channels.append(_validate_and_resize(l_band_real, reference_shape, "L_real"))
    channel_names.append("L_band_real")
    channels.append(_validate_and_resize(l_band_imag, reference_shape, "L_imag"))
    channel_names.append("L_band_imag")

    # S-band (optional)
    if s_band_real is not None and s_band_imag is not None:
        channels.append(_validate_and_resize(s_band_real, reference_shape, "S_real"))
        channel_names.append("S_band_real")
        channels.append(_validate_and_resize(s_band_imag, reference_shape, "S_imag"))
        channel_names.append("S_band_imag")

    # Topographic layers
    if dem is not None:
        channels.append(_validate_and_resize(dem, reference_shape, "DEM"))
        channel_names.append("DEM_elevation")

    if slope is not None:
        channels.append(_validate_and_resize(slope, reference_shape, "Slope"))
        channel_names.append("Slope_degrees")

    if aspect is not None:
        channels.append(_validate_and_resize(aspect, reference_shape, "Aspect"))
        channel_names.append("Aspect_degrees")

    # Stack: (C, H, W)
    tensor = np.stack(channels, axis=0).astype(np.float32)

    # Normalize per channel (z-score)
    if normalize:
        tensor = _normalize_channels(tensor)

    logger.info(
        f"Built tensor: shape={tensor.shape}, "
        f"channels={channel_names}, "
        f"dtype={tensor.dtype}"
    )

    return tensor


def _validate_and_resize(
    array: np.ndarray,
    target_shape: tuple[int, int],
    name: str,
) -> np.ndarray:
    """Validate and resize array to match target shape.

    Args:
        array: Input 2D array.
        target_shape: Expected (H, W) shape.
        name: Channel name for logging.

    Returns:
        Array resized/validated to target_shape.
    """
    if array.ndim != 2:
        raise ValueError(f"{name}: expected 2D array, got shape {array.shape}")

    if array.shape != target_shape:
        logger.warning(
            f"{name}: shape {array.shape} != target {target_shape}, "
            f"resizing with bilinear interpolation"
        )
        from scipy.ndimage import zoom

        zoom_factors = (
            target_shape[0] / array.shape[0],
            target_shape[1] / array.shape[1],
        )
        array = zoom(array.astype(np.float64), zoom_factors, order=1)

    return array.astype(np.float32)


def _normalize_channels(tensor: np.ndarray) -> np.ndarray:
    """Normalize each channel to zero-mean, unit-variance.

    NaN values are excluded from statistics and preserved in output.

    Args:
        tensor: Array of shape (C, H, W).

    Returns:
        Normalized tensor.
    """
    normalized = tensor.copy()
    for ch in range(tensor.shape[0]):
        channel = tensor[ch]
        valid = channel[np.isfinite(channel)]
        if len(valid) > 0:
            mean = np.mean(valid)
            std = np.std(valid)
            if std > 1e-8:
                normalized[ch] = (channel - mean) / std
            else:
                normalized[ch] = channel - mean
    return normalized


def save_coregistered_tensor(
    tensor: np.ndarray,
    output_path: str | Path,
    channel_names: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
    use_h5: bool = False,
) -> Path:
    """Save the co-registered tensor to disk.

    Args:
        tensor: Multi-channel tensor (C, H, W).
        output_path: Output file path.
        channel_names: Optional list of channel names.
        metadata: Additional metadata to include.
        use_h5: If True, save as HDF5; otherwise, NumPy .npy.

    Returns:
        Path to saved file.
    """
    meta = metadata or {}
    meta.update({
        'shape': list(tensor.shape),
        'dtype': str(tensor.dtype),
        'num_channels': tensor.shape[0],
    })
    if channel_names:
        meta['channel_names'] = channel_names

    if use_h5:
        return save_tensor_h5(tensor, output_path, metadata=meta)
    else:
        return save_tensor(tensor, output_path, metadata=meta)


def load_complex_binary(
    path: str | Path,
    shape: tuple[int, int],
    dtype: str = "<c8",
    offset: int = 0,
) -> np.ndarray:
    """Load a complex-valued binary array from a raw .img file.

    Args:
        path: Path to the binary file.
        shape: (rows, cols) expected shape.
        dtype: NumPy dtype string (e.g., "<c8" for complex64 little-endian).
        offset: Byte offset to start reading from.

    Returns:
        Complex-valued 2D array.
    """
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.dtype(dtype), offset=offset)

    expected_size = shape[0] * shape[1]
    if data.size < expected_size:
        raise ValueError(
            f"File {path} contains {data.size} elements, "
            f"expected {expected_size} for shape {shape}"
        )

    data = data[:expected_size].reshape(shape)
    logger.info(f"Loaded complex binary: {path} → {data.shape} ({data.dtype})")
    return data
