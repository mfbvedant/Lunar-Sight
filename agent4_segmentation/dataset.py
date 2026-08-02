"""
Segmentation Dataset
=====================
Multi-channel PyTorch dataset with NaN masking for weakly-supervised training.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class LunarSegDataset(Dataset):
    """Multi-channel segmentation dataset with NaN-masked labels.

    Args:
        feature_tensor_path: Path to polarimetric feature tensor (C, H, W).
        label_map_path: Path to pseudo-label map (H, W) with NaN.
        topo_paths: Optional dict of {name: path} for DEM/slope channels.
        patch_size: Size of extracted patches.
        stride: Stride between patches.
        augment: Whether to apply augmentation.
    """

    def __init__(
        self,
        feature_tensor_path: str | Path,
        label_map_path: str | Path,
        topo_paths: dict[str, str | Path] | None = None,
        patch_size: int = 256,
        stride: int = 128,
        augment: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.augment = augment

        # Load feature tensor
        self.features = np.load(str(feature_tensor_path), allow_pickle=False)
        if self.features.ndim == 2:
            self.features = self.features[np.newaxis]

        # Load label map
        self.labels = np.load(str(label_map_path), allow_pickle=False)

        # Append topographic channels
        if topo_paths:
            topo_channels = []
            for name, path in topo_paths.items():
                arr = np.load(str(path), allow_pickle=False)
                if arr.shape != self.labels.shape:
                    from scipy.ndimage import zoom
                    factors = (
                        self.labels.shape[0] / arr.shape[0],
                        self.labels.shape[1] / arr.shape[1],
                    )
                    arr = zoom(arr, factors, order=1)
                topo_channels.append(arr)
                logger.info(f"Added topo channel '{name}': {arr.shape}")

            if topo_channels:
                topo_stack = np.stack(topo_channels, axis=0).astype(np.float32)
                self.features = np.concatenate([self.features, topo_stack], axis=0)

        self.n_channels = self.features.shape[0]
        self.height = self.features.shape[1]
        self.width = self.features.shape[2]

        # Create valid mask (where labels are 0 or 1, not NaN)
        self.valid_mask = np.isfinite(self.labels)

        # Compute patch grid (only patches with at least some valid labels)
        self.patches = self._compute_patches()
        logger.info(
            f"Dataset: {self.n_channels} channels, {len(self.patches)} valid patches, "
            f"patch_size={patch_size}, stride={stride}"
        )

    def _compute_patches(self) -> list[tuple[int, int]]:
        """Compute patch coordinates, filtering for patches with valid labels."""
        patches = []
        min_valid_fraction = 0.01  # At least 1% of pixels must be labeled

        for y in range(0, self.height - self.patch_size + 1, self.stride):
            for x in range(0, self.width - self.patch_size + 1, self.stride):
                label_patch = self.labels[y:y+self.patch_size, x:x+self.patch_size]
                valid_fraction = np.isfinite(label_patch).mean()
                if valid_fraction >= min_valid_fraction:
                    patches.append((y, x))

        return patches

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a patch with features, labels, and valid mask.

        Returns:
            Tuple of:
                - features: (C, H, W) float tensor.
                - labels: (H, W) long tensor (0=rock, 1=ice, 0 for NaN).
                - valid_mask: (H, W) bool tensor (True where label is valid).
        """
        y, x = self.patches[idx]

        feat = self.features[:, y:y+self.patch_size, x:x+self.patch_size].copy()
        lbl = self.labels[y:y+self.patch_size, x:x+self.patch_size].copy()

        # Create valid mask before converting NaN to 0
        valid = np.isfinite(lbl)
        lbl = np.nan_to_num(lbl, nan=0.0)

        if self.augment:
            feat, lbl, valid = self._augment(feat, lbl, valid)

        feat_tensor = torch.from_numpy(feat).float()
        lbl_tensor = torch.from_numpy(lbl).long()
        valid_tensor = torch.from_numpy(valid.astype(np.float32))

        return feat_tensor, lbl_tensor, valid_tensor

    def _augment(self, feat, lbl, valid):
        """Apply random augmentations."""
        k = np.random.randint(0, 4)
        if k > 0:
            feat = np.rot90(feat, k, axes=(1, 2)).copy()
            lbl = np.rot90(lbl, k).copy()
            valid = np.rot90(valid, k).copy()

        if np.random.rand() > 0.5:
            feat = np.flip(feat, axis=2).copy()
            lbl = np.flip(lbl, axis=1).copy()
            valid = np.flip(valid, axis=1).copy()

        if np.random.rand() > 0.5:
            feat = np.flip(feat, axis=1).copy()
            lbl = np.flip(lbl, axis=0).copy()
            valid = np.flip(valid, axis=0).copy()

        return feat, lbl, valid
