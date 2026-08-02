"""
Despeckling Dataset
====================
Patch-based PyTorch dataset for training the CV-CNN on complex covariance data.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class DespecklingDataset(Dataset):
    """Patch-based dataset for CV-CNN despeckling training.

    Extracts overlapping tiles from a co-registered tensor and converts
    them to complex64 PyTorch tensors for self-supervised denoising.

    The training is self-supervised: the noisy input IS the target
    (autoencoder learns to reconstruct clean signal from noisy input).
    Multi-look averaging can provide cleaner targets if available.

    Args:
        tensor_path: Path to the co-registered tensor (.npy).
        patch_size: Size of extracted patches (square).
        stride: Stride between patches (controls overlap).
        transform: Optional augmentation transforms.
    """

    def __init__(
        self,
        tensor_path: str | Path,
        patch_size: int = 128,
        stride: int = 64,
        transform: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform

        # Load tensor
        tensor_path = Path(tensor_path)
        self.data = np.load(str(tensor_path), allow_pickle=False)
        logger.info(f"Loaded tensor: {self.data.shape} from {tensor_path}")

        # Ensure (C, H, W) format
        if self.data.ndim == 2:
            self.data = self.data[np.newaxis, :, :]

        self.n_channels, self.height, self.width = self.data.shape

        # Compute patch grid
        self.patches = self._compute_patch_grid()
        logger.info(f"Created {len(self.patches)} patches ({patch_size}×{patch_size}, stride={stride})")

    def _compute_patch_grid(self) -> list[tuple[int, int]]:
        """Compute top-left corner coordinates for all patches."""
        patches = []
        for y in range(0, self.height - self.patch_size + 1, self.stride):
            for x in range(0, self.width - self.patch_size + 1, self.stride):
                patches.append((y, x))
        return patches

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a patch pair (input, target).

        For self-supervised despeckling, input == target (the network
        learns to denoise by reconstructing the input).

        Returns:
            Tuple of (input_tensor, target_tensor), both complex64.
        """
        y, x = self.patches[idx]
        patch = self.data[
            :,
            y:y + self.patch_size,
            x:x + self.patch_size,
        ].copy()

        # Apply augmentation
        if self.transform:
            patch = self._augment(patch)

        # Convert to complex tensor
        # Channels are packed as [Re(C11), Re(C12), Im(C12), Re(C22)]
        # We keep them as real-valued for the network (it handles complex internally)
        tensor = torch.from_numpy(patch).float()

        return tensor, tensor.clone()

    def _augment(self, patch: np.ndarray) -> np.ndarray:
        """Apply random augmentations (90° rotations, flips)."""
        # Random 90° rotation
        k = np.random.randint(0, 4)
        if k > 0:
            patch = np.rot90(patch, k, axes=(1, 2)).copy()

        # Random horizontal flip
        if np.random.rand() > 0.5:
            patch = np.flip(patch, axis=2).copy()

        # Random vertical flip
        if np.random.rand() > 0.5:
            patch = np.flip(patch, axis=1).copy()

        return patch


def create_data_loaders(
    tensor_path: str | Path,
    patch_size: int = 128,
    stride: int = 64,
    batch_size: int = 8,
    train_split: float = 0.8,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> tuple:
    """Create train and validation data loaders.

    Args:
        tensor_path: Path to co-registered tensor.
        patch_size: Patch size.
        stride: Patch extraction stride.
        batch_size: Training batch size.
        train_split: Fraction of patches for training.
        num_workers: DataLoader workers.
        pin_memory: Pin memory for GPU transfer.

    Returns:
        Tuple of (train_loader, val_loader, dataset).
    """
    from torch.utils.data import DataLoader, random_split

    dataset = DespecklingDataset(
        tensor_path, patch_size, stride, transform=True,
    )

    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    logger.info(f"Train: {train_size} patches, Val: {val_size} patches")
    return train_loader, val_loader, dataset
