"""
Despeckling Inference
======================
Run trained CV-CNN model on full scene with tile-based inference
and Hann window blending to avoid boundary artifacts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def run_inference(
    model: nn.Module,
    input_tensor: np.ndarray,
    tile_size: int = 128,
    overlap: int = 32,
    device: str = "auto",
    batch_size: int = 4,
) -> np.ndarray:
    """Run despeckling inference on full scene with tiled processing.

    Uses overlapping tiles with Hann window blending to eliminate
    boundary artifacts from convolution edge effects.

    Args:
        model: Trained CV-CNN model.
        input_tensor: Input tensor (C, H, W), dtype=float32.
        tile_size: Size of processing tiles (square).
        overlap: Overlap between adjacent tiles.
        device: Device for inference.
        batch_size: Number of tiles to process in parallel.

    Returns:
        Despeckled tensor (C, H, W), same shape as input.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    n_channels, height, width = input_tensor.shape
    stride = tile_size - overlap

    # Create Hann blending window
    hann_1d = np.hanning(tile_size)
    hann_2d = np.outer(hann_1d, hann_1d).astype(np.float32)

    # Output accumulator and weight map
    output = np.zeros_like(input_tensor, dtype=np.float64)
    weight_map = np.zeros((height, width), dtype=np.float64)

    # Collect tile coordinates
    tiles = []
    for y in range(0, height - tile_size + 1, stride):
        for x in range(0, width - tile_size + 1, stride):
            tiles.append((y, x))
    # Handle right/bottom edges
    if tiles and tiles[-1][0] + tile_size < height:
        for x in range(0, width - tile_size + 1, stride):
            tiles.append((height - tile_size, x))
    if tiles and tiles[-1][1] + tile_size < width:
        for y in range(0, height - tile_size + 1, stride):
            tiles.append((y, width - tile_size))

    logger.info(f"Running inference on {len(tiles)} tiles ({tile_size}×{tile_size}, overlap={overlap})")

    # Process in batches
    with torch.no_grad():
        for batch_start in range(0, len(tiles), batch_size):
            batch_tiles = tiles[batch_start:batch_start + batch_size]

            # Extract batch
            batch = np.stack([
                input_tensor[:, y:y+tile_size, x:x+tile_size]
                for y, x in batch_tiles
            ])

            # Forward pass
            batch_tensor = torch.from_numpy(batch).float().to(device)
            batch_complex = torch.complex(batch_tensor, torch.zeros_like(batch_tensor))
            pred = model(batch_complex)
            pred_np = pred.real.cpu().numpy()

            # Accumulate with Hann blending
            for i, (y, x) in enumerate(batch_tiles):
                for ch in range(n_channels):
                    output[ch, y:y+tile_size, x:x+tile_size] += pred_np[i, ch] * hann_2d
                weight_map[y:y+tile_size, x:x+tile_size] += hann_2d

            if (batch_start // batch_size) % 20 == 0:
                pct = min(100, (batch_start + batch_size) / len(tiles) * 100)
                logger.info(f"  Progress: {pct:.1f}%")

    # Normalize by weight map
    valid = weight_map > 1e-8
    for ch in range(n_channels):
        output[ch][valid] /= weight_map[valid]
        # Fill any uncovered pixels with input values
        output[ch][~valid] = input_tensor[ch][~valid]

    logger.info("Inference complete")
    return output.astype(np.float32)


def verify_phase_integrity(
    original: np.ndarray,
    despeckled: np.ndarray,
    tolerance_deg: float = 10.0,
) -> dict[str, float]:
    """Verify that despeckling preserved off-diagonal phase angles.

    Compares phase angles of complex cross-terms before and after despeckling.

    Args:
        original: Original tensor (C, H, W).
        despeckled: Despeckled tensor (C, H, W).
        tolerance_deg: Maximum acceptable phase deviation.

    Returns:
        Dict with phase integrity statistics.
    """
    if original.shape[0] < 3:
        return {"status": "skipped", "reason": "fewer than 3 channels"}

    # Reconstruct complex cross-term from channels 1,2 (Re, Im of C₁₂)
    orig_phase = np.arctan2(original[2], original[1])  # atan2(Im, Re)
    desp_phase = np.arctan2(despeckled[2], despeckled[1])

    # Phase difference
    phase_diff = np.degrees(np.abs(orig_phase - desp_phase))
    phase_diff = np.minimum(phase_diff, 360 - phase_diff)  # Wrap around

    valid = np.isfinite(phase_diff)

    stats = {
        "mean_phase_diff_deg": float(np.mean(phase_diff[valid])),
        "max_phase_diff_deg": float(np.max(phase_diff[valid])),
        "std_phase_diff_deg": float(np.std(phase_diff[valid])),
        "pct_within_tolerance": float(
            np.sum(phase_diff[valid] < tolerance_deg) / np.sum(valid) * 100
        ),
        "tolerance_deg": tolerance_deg,
        "passed": bool(np.mean(phase_diff[valid]) < tolerance_deg),
    }

    if stats["passed"]:
        logger.info(f"Phase integrity CHECK PASSED: mean diff = {stats['mean_phase_diff_deg']:.2f}°")
    else:
        logger.warning(f"Phase integrity CHECK FAILED: mean diff = {stats['mean_phase_diff_deg']:.2f}°")

    return stats
