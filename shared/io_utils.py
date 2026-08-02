"""
LunarSight I/O Utilities
=========================
Tensor save/load helpers, Google Drive mount, and checkpoint management.
"""

from __future__ import annotations

import json
import os
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# Tensor I/O (NumPy)
# ============================================================

def save_tensor(
    array: np.ndarray,
    path: str | Path,
    metadata: Optional[dict[str, Any]] = None,
) -> Path:
    """Save a NumPy array to disk with optional JSON sidecar metadata.

    Args:
        array: The NumPy array to save.
        path: Output file path (will use .npy extension).
        metadata: Optional dict to save as a JSON sidecar file.

    Returns:
        Path to the saved .npy file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure .npy extension
    if path.suffix != '.npy':
        path = path.with_suffix('.npy')

    np.save(str(path), array)
    logger.info(f"Saved tensor {array.shape} ({array.dtype}) → {path}")

    if metadata is not None:
        meta_path = path.with_suffix('.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Saved metadata → {meta_path}")

    return path


def load_tensor(path: str | Path) -> np.ndarray:
    """Load a NumPy array from disk.

    Args:
        path: Path to .npy file.

    Returns:
        The loaded NumPy array.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Tensor file not found: {path}")

    array = np.load(str(path), allow_pickle=False)
    logger.info(f"Loaded tensor {array.shape} ({array.dtype}) ← {path}")
    return array


def load_metadata(path: str | Path) -> dict[str, Any]:
    """Load the JSON sidecar metadata for a tensor.

    Args:
        path: Path to .npy file (will look for .json with same stem).

    Returns:
        Metadata dictionary.
    """
    path = Path(path)
    meta_path = path.with_suffix('.json')
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    with open(meta_path, 'r') as f:
        return json.load(f)


# ============================================================
# HDF5 I/O (for large multi-channel tensors)
# ============================================================

def save_tensor_h5(
    array: np.ndarray,
    path: str | Path,
    dataset_name: str = "data",
    metadata: Optional[dict[str, Any]] = None,
    compression: str = "gzip",
) -> Path:
    """Save a NumPy array to HDF5 format with optional compression.

    Args:
        array: The NumPy array to save.
        path: Output file path (will use .h5 extension).
        dataset_name: Name of the dataset inside the HDF5 file.
        metadata: Optional dict stored as HDF5 attributes.
        compression: Compression algorithm (default: gzip).

    Returns:
        Path to the saved .h5 file.
    """
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix not in ('.h5', '.hdf5'):
        path = path.with_suffix('.h5')

    with h5py.File(str(path), 'w') as hf:
        ds = hf.create_dataset(
            dataset_name, data=array, compression=compression
        )
        if metadata:
            for k, v in metadata.items():
                ds.attrs[k] = str(v) if not isinstance(v, (int, float)) else v

    logger.info(f"Saved HDF5 tensor {array.shape} → {path}")
    return path


def load_tensor_h5(
    path: str | Path,
    dataset_name: str = "data",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a NumPy array and metadata from HDF5 format.

    Args:
        path: Path to .h5 file.
        dataset_name: Name of the dataset inside the HDF5 file.

    Returns:
        Tuple of (array, metadata_dict).
    """
    import h5py

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")

    with h5py.File(str(path), 'r') as hf:
        array = hf[dataset_name][:]
        metadata = dict(hf[dataset_name].attrs)

    logger.info(f"Loaded HDF5 tensor {array.shape} ← {path}")
    return array, metadata


# ============================================================
# Google Drive / Colab Helpers
# ============================================================

def mount_google_drive(mount_point: str = "/content/drive") -> Path:
    """Mount Google Drive in a Colab environment.

    Args:
        mount_point: Mount path (default Colab convention).

    Returns:
        Path to the mounted Drive root (MyDrive).

    Raises:
        EnvironmentError: If not running in Google Colab.
    """
    try:
        from google.colab import drive  # type: ignore
        drive.mount(mount_point, force_remount=False)
        drive_root = Path(mount_point) / "MyDrive"
        logger.info(f"Google Drive mounted at {drive_root}")
        return drive_root
    except ImportError:
        raise EnvironmentError(
            "google.colab not available — not running in Colab environment. "
            "Use local paths instead."
        )


def ensure_drive_dir(path: str | Path) -> Path:
    """Create a directory on Google Drive (or locally), return the Path.

    Args:
        path: Directory path to create.

    Returns:
        The created Path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_colab() -> bool:
    """Check if the current environment is Google Colab."""
    try:
        import google.colab  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


# ============================================================
# Checkpoint Helpers
# ============================================================

def find_latest_checkpoint(
    checkpoint_dir: str | Path,
    prefix: str = "checkpoint",
    extension: str = ".pt",
) -> Optional[Path]:
    """Find the latest checkpoint file in a directory by modification time.

    Args:
        checkpoint_dir: Directory to search.
        prefix: Filename prefix filter.
        extension: File extension filter.

    Returns:
        Path to latest checkpoint, or None if no checkpoints found.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoints = sorted(
        checkpoint_dir.glob(f"{prefix}*{extension}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if checkpoints:
        logger.info(f"Found latest checkpoint: {checkpoints[0]}")
        return checkpoints[0]
    return None


def get_output_dir(config: dict, subdir: str = "") -> Path:
    """Get the output directory from config, creating it if needed.

    Args:
        config: Mission config dict.
        subdir: Optional subdirectory name.

    Returns:
        Path to output directory.
    """
    base = Path(config.get("data", {}).get("output_dir", "./output"))
    out = base / subdir if subdir else base
    out.mkdir(parents=True, exist_ok=True)
    return out
