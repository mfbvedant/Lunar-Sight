"""
CV-CNN Despeckling Model
=========================
Complex-Valued Convolutional Neural Network autoencoder for SAR
speckle noise suppression while preserving polarimetric phase.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from agent2_despeckling.complex_layers import (
    ComplexConv2d,
    ComplexBatchNorm,
    ComplexReLU,
    ComplexDropout,
    ComplexAvgPool2d,
    ComplexConvTranspose2d,
)


class ComplexEncoderBlock(nn.Module):
    """Encoder block: ComplexConv → ComplexBN → ComplexReLU → Pool."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = ComplexConv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = ComplexBatchNorm(out_ch)
        self.relu1 = ComplexReLU()
        self.conv2 = ComplexConv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = ComplexBatchNorm(out_ch)
        self.relu2 = ComplexReLU()
        self.pool = ComplexAvgPool2d(kernel_size=2, stride=2)
        self.dropout = ComplexDropout(p=dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (pooled, skip_connection)."""
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        skip = x
        x = self.pool(x)
        x = self.dropout(x)
        return x, skip


class ComplexDecoderBlock(nn.Module):
    """Decoder block: Upsample → Concat skip → ComplexConv → ComplexBN → ComplexReLU."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.up = ComplexConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv1 = ComplexConv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = ComplexBatchNorm(out_ch)
        self.relu1 = ComplexReLU()
        self.conv2 = ComplexConv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = ComplexBatchNorm(out_ch)
        self.relu2 = ComplexReLU()
        self.dropout = ComplexDropout(p=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle size mismatch from pooling
        if x.shape != skip.shape:
            diff_h = skip.shape[2] - x.shape[2]
            diff_w = skip.shape[3] - x.shape[3]
            x = nn.functional.pad(x.real, [0, diff_w, 0, diff_h]) + \
                1j * nn.functional.pad(x.imag, [0, diff_w, 0, diff_h])
        x = torch.cat([x, skip], dim=1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        return x


class CVCNN(nn.Module):
    """Complex-Valued CNN Denoising Autoencoder.

    Architecture:
        - Encoder: 4 blocks of (ComplexConv → BN → ReLU → Pool)
        - Bottleneck: ComplexConv block
        - Decoder: 4 blocks of (Upsample → Skip → ComplexConv → BN → ReLU)
        - Output: ComplexConv to match input channels

    All operations in the complex domain.

    Args:
        in_channels: Number of input channels (4 for C₂, 9 for C₃ — real-packed).
        base_filters: Number of filters in first encoder layer.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_filters: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        bf = base_filters

        # Encoder
        self.enc1 = ComplexEncoderBlock(in_channels, bf, dropout)
        self.enc2 = ComplexEncoderBlock(bf, bf * 2, dropout)
        self.enc3 = ComplexEncoderBlock(bf * 2, bf * 4, dropout)
        self.enc4 = ComplexEncoderBlock(bf * 4, bf * 8, dropout)

        # Bottleneck
        self.bottleneck_conv1 = ComplexConv2d(bf * 8, bf * 16, kernel_size=3, padding=1)
        self.bottleneck_bn1 = ComplexBatchNorm(bf * 16)
        self.bottleneck_relu1 = ComplexReLU()
        self.bottleneck_conv2 = ComplexConv2d(bf * 16, bf * 16, kernel_size=3, padding=1)
        self.bottleneck_bn2 = ComplexBatchNorm(bf * 16)
        self.bottleneck_relu2 = ComplexReLU()

        # Decoder
        self.dec4 = ComplexDecoderBlock(bf * 16, bf * 8, bf * 8, dropout)
        self.dec3 = ComplexDecoderBlock(bf * 8, bf * 4, bf * 4, dropout)
        self.dec2 = ComplexDecoderBlock(bf * 4, bf * 2, bf * 2, dropout)
        self.dec1 = ComplexDecoderBlock(bf * 2, bf, bf, dropout)

        # Output
        self.out_conv = ComplexConv2d(bf, in_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Complex tensor (B, C, H, W), dtype=complex64.

        Returns:
            Despeckled complex tensor, same shape as input.
        """
        # Encode
        x1, skip1 = self.enc1(x)
        x2, skip2 = self.enc2(x1)
        x3, skip3 = self.enc3(x2)
        x4, skip4 = self.enc4(x3)

        # Bottleneck
        b = self.bottleneck_relu1(self.bottleneck_bn1(self.bottleneck_conv1(x4)))
        b = self.bottleneck_relu2(self.bottleneck_bn2(self.bottleneck_conv2(b)))

        # Decode
        d4 = self.dec4(b, skip4)
        d3 = self.dec3(d4, skip3)
        d2 = self.dec2(d3, skip2)
        d1 = self.dec1(d2, skip1)

        # Output
        out = self.out_conv(d1)
        return out


def create_model(
    in_channels: int = 4,
    base_filters: int = 32,
    dropout: float = 0.1,
    device: Optional[str] = None,
) -> CVCNN:
    """Factory function to create and initialize the CV-CNN model.

    Args:
        in_channels: Input channels (4 for C₂ real-packed).
        base_filters: Base filter count.
        dropout: Dropout rate.
        device: Target device ("cuda", "cpu", or None for auto).

    Returns:
        Initialized CVCNN model on the specified device.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CVCNN(
        in_channels=in_channels,
        base_filters=base_filters,
        dropout=dropout,
    )

    # Initialize weights
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CV-CNN model: {total_params:,} params ({trainable:,} trainable) on {device}")

    return model
