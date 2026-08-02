"""
Segmentation Inference
========================
Full-scene U-Net inference → Binary Ice Mask + Confidence Map.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def run_segmentation_inference(
    model: nn.Module,
    feature_tensor: np.ndarray,
    tile_size: int = 256,
    overlap: int = 64,
    device: str = "auto",
    batch_size: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Run U-Net inference on full scene with tiled processing.

    Args:
        model: Trained U-Net model.
        feature_tensor: Input (C, H, W) float32.
        tile_size: Tile size for processing.
        overlap: Overlap between tiles.
        device: Device.
        batch_size: Batch size for tiles.

    Returns:
        Tuple of:
            - ice_mask: Binary mask (H, W), 0=rock, 1=ice.
            - confidence: Confidence map (H, W), [0, 1].
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    n_ch, height, width = feature_tensor.shape
    stride = tile_size - overlap

    # Accumulate softmax probabilities
    prob_sum = np.zeros((2, height, width), dtype=np.float64)
    weight_map = np.zeros((height, width), dtype=np.float64)

    hann_1d = np.hanning(tile_size)
    hann_2d = np.outer(hann_1d, hann_1d).astype(np.float32)

    tiles = []
    for y in range(0, height - tile_size + 1, stride):
        for x in range(0, width - tile_size + 1, stride):
            tiles.append((y, x))

    logger.info(f"Segmentation inference: {len(tiles)} tiles")

    with torch.no_grad():
        for i in range(0, len(tiles), batch_size):
            batch_tiles = tiles[i:i+batch_size]
            batch = np.stack([
                feature_tensor[:, y:y+tile_size, x:x+tile_size]
                for y, x in batch_tiles
            ])

            batch_t = torch.from_numpy(batch).float().to(device)
            logits = model(batch_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            for j, (y, x) in enumerate(batch_tiles):
                for c in range(2):
                    prob_sum[c, y:y+tile_size, x:x+tile_size] += probs[j, c] * hann_2d
                weight_map[y:y+tile_size, x:x+tile_size] += hann_2d

    # Normalize
    valid = weight_map > 1e-8
    for c in range(2):
        prob_sum[c][valid] /= weight_map[valid]

    # Ice mask: argmax of class probabilities
    ice_mask = np.argmax(prob_sum, axis=0).astype(np.uint8)

    # Confidence: max probability
    confidence = np.max(prob_sum, axis=0).astype(np.float32)
    confidence[~valid] = 0.0

    logger.info(
        f"Ice mask: {np.sum(ice_mask == 1)} ice pixels, "
        f"{np.sum(ice_mask == 0)} rock pixels"
    )

    return ice_mask, confidence
