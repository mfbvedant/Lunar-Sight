"""
Segmentation Model
===================
U-Net wrapper using SMP (segmentation-models-pytorch) with custom
input channels for multi-spectral/polarimetric data.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def create_unet(
    in_channels: int,
    num_classes: int = 2,
    encoder_name: str = "efficientnet-b3",
    encoder_weights: str = "imagenet",
    device: Optional[str] = None,
) -> nn.Module:
    """Create a U-Net segmentation model using SMP.

    Handles the case where in_channels > 3 by initializing the first
    conv layer specially: first 3 channels get ImageNet weights, rest
    get Kaiming initialization.

    Args:
        in_channels: Number of input channels (from feature tensor + topo).
        num_classes: Number of output classes (2 = ice vs rock).
        encoder_name: SMP encoder backbone.
        encoder_weights: Pretrained weights ("imagenet" or None).
        device: Target device.

    Returns:
        Initialized U-Net model.
    """
    import segmentation_models_pytorch as smp

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights if in_channels <= 3 else None,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,  # Raw logits
    )

    # If in_channels > 3 and we want ImageNet pretrained:
    # Load a 3-channel pretrained model and transfer weights
    if in_channels > 3 and encoder_weights:
        _transfer_encoder_weights(model, encoder_name, encoder_weights, in_channels)

    model = model.to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"U-Net ({encoder_name}): {total:,} params ({trainable:,} trainable)")
    logger.info(f"Input channels: {in_channels}, Output classes: {num_classes}")

    return model


def _transfer_encoder_weights(model, encoder_name, encoder_weights, in_channels):
    """Transfer ImageNet weights from a 3-channel model to N-channel model.

    Strategy: copy first 3 channel weights, Kaiming-init the rest.
    """
    import segmentation_models_pytorch as smp

    try:
        pretrained = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=2,
        )

        # Find the first conv layer
        for name, module in model.encoder.named_modules():
            if isinstance(module, nn.Conv2d) and module.in_channels == in_channels:
                # Find corresponding layer in pretrained
                for p_name, p_module in pretrained.encoder.named_modules():
                    if p_name == name and isinstance(p_module, nn.Conv2d):
                        with torch.no_grad():
                            # Copy first 3 channels
                            module.weight[:, :3] = p_module.weight
                            # Kaiming init for remaining channels
                            if in_channels > 3:
                                nn.init.kaiming_normal_(
                                    module.weight[:, 3:],
                                    mode='fan_out', nonlinearity='relu'
                                )
                            if module.bias is not None and p_module.bias is not None:
                                module.bias.copy_(p_module.bias)
                        logger.info(
                            f"Transferred ImageNet weights to {name}: "
                            f"3/{in_channels} channels from pretrained"
                        )
                        return

        logger.warning("Could not find first conv layer for weight transfer")

    except Exception as e:
        logger.warning(f"Weight transfer failed: {e}. Using random init.")
