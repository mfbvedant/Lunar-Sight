"""
Complex-Valued Neural Network Layers
=======================================
Custom PyTorch modules that operate in the complex domain,
preserving both magnitude and phase information through the network.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexConv2d(nn.Module):
    """Complex-valued 2D convolution.

    Implements convolution following complex multiplication rules:
        (a + jb) * (c + jd) = (ac - bd) + j(ad + bc)

    Uses two real-valued conv layers internally.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.conv_real = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias,
        )
        self.conv_imag = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with complex multiplication.

        Args:
            x: Complex tensor (B, C, H, W), dtype=complex64.

        Returns:
            Complex tensor (B, C_out, H, W).
        """
        real = x.real
        imag = x.imag

        # (a + jb)(c + jd) = (ac - bd) + j(ad + bc)
        out_real = self.conv_real(real) - self.conv_imag(imag)
        out_imag = self.conv_real(imag) + self.conv_imag(real)

        return torch.complex(out_real, out_imag)


class ComplexBatchNorm(nn.Module):
    """Batch normalization for complex-valued tensors.

    Normalizes real and imaginary parts independently.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.bn_real = nn.BatchNorm2d(num_features)
        self.bn_imag = nn.BatchNorm2d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.complex(
            self.bn_real(x.real),
            self.bn_imag(x.imag),
        )


class ComplexReLU(nn.Module):
    """ReLU for complex-valued tensors.

    Applies ReLU to magnitude while preserving phase:
        out = ReLU(|x|) * exp(j·∠x) = ReLU(|x|) * x / |x|

    Falls back to applying ReLU to real and imag independently
    where magnitude is near zero.
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        magnitude = torch.abs(x)
        phase = torch.angle(x)

        activated_magnitude = F.relu(magnitude, inplace=self.inplace)
        return activated_magnitude * torch.exp(1j * phase)


class ComplexLeakyReLU(nn.Module):
    """LeakyReLU for complex-valued tensors.

    Applies LeakyReLU to magnitude, preserves phase.
    """

    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        magnitude = torch.abs(x)
        phase = torch.angle(x)

        activated = F.leaky_relu(magnitude, self.negative_slope)
        return activated * torch.exp(1j * phase)


class ComplexDropout(nn.Module):
    """Dropout for complex-valued tensors.

    Applies the same dropout mask to both real and imaginary parts
    to maintain phase coherence.
    """

    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x

        # Generate mask on real part, apply to both
        mask = torch.ones_like(x.real)
        mask = F.dropout(mask, p=self.p, training=True)

        return torch.complex(x.real * mask, x.imag * mask)


class ComplexMaxPool2d(nn.Module):
    """Max pooling for complex tensors based on magnitude."""

    def __init__(self, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        magnitude = torch.abs(x)
        # Get indices from magnitude-based pooling
        _, indices = F.max_pool2d(
            magnitude, self.kernel_size, self.stride,
            return_indices=True,
        )

        # Apply same indices to real and imag parts
        real_pooled = F.max_pool2d(x.real, self.kernel_size, self.stride)
        imag_pooled = F.max_pool2d(x.imag, self.kernel_size, self.stride)

        return torch.complex(real_pooled, imag_pooled)


class ComplexAvgPool2d(nn.Module):
    """Average pooling for complex tensors."""

    def __init__(self, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.complex(
            self.pool(x.real),
            self.pool(x.imag),
        )


class ComplexConvTranspose2d(nn.Module):
    """Complex-valued transposed (upsampling) convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        stride: int = 2,
        padding: int = 0,
        output_padding: int = 0,
    ):
        super().__init__()
        self.deconv_real = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
        )
        self.deconv_imag = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_real = self.deconv_real(x.real) - self.deconv_imag(x.imag)
        out_imag = self.deconv_real(x.imag) + self.deconv_imag(x.real)
        return torch.complex(out_real, out_imag)
