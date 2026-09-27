"""
models/unet.py

Attention U-Net implementation for lung segmentation.

Architecture:
  - Encoder: 4-level feature extraction with MaxPool downsampling
  - Bridge: Bottleneck convolution block
  - Decoder: 4-level upsampling with attention gates and skip connections
  - Output: 1×1 Conv → Sigmoid for binary mask

Total parameters: ~3-4M depending on feature sizes.
Memory during training (fp32, batch=8, 256×256): ~2-3 GB.

Reference:
  Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas",
  MIDL 2018.
"""

import logging
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DoubleConv(nn.Module):
    """
    Two sequential Conv2d → BatchNorm2d → ReLU blocks.

    Standard building block for U-Net encoder and decoder.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.insert(3, nn.Dropout2d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """
    Soft attention gate for skip connections.

    Learns to focus on relevant regions by gating encoder features
    using the decoder's upsampled feature map as a guide signal.

    Args:
        F_g: Number of channels in gate signal (from decoder).
        F_l: Number of channels in skip connection (from encoder).
        F_int: Number of intermediate channels.
    """

    def __init__(self, F_g: int, F_l: int, F_int: int) -> None:
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self, g: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            g: Gate signal from decoder, shape (B, F_g, H, W).
            x: Skip connection from encoder, shape (B, F_l, H, W).

        Returns:
            Attention-gated skip connection, shape (B, F_l, H, W).
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class EncoderBlock(nn.Module):
    """Encoder block: DoubleConv + MaxPool downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            skip: Feature map before pooling (for skip connection).
            pooled: Downsampled feature map.
        """
        skip = self.conv(x)
        pooled = self.pool(skip)
        return skip, pooled


class DecoderBlock(nn.Module):
    """
    Decoder block: Upsample + AttentionGate + skip connection + DoubleConv.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.attention = AttentionGate(
            F_g=in_channels // 2,
            F_l=skip_channels,
            F_int=skip_channels // 2,
        )
        self.conv = DoubleConv(
            in_channels // 2 + skip_channels, out_channels, dropout=dropout
        )

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor
    ) -> torch.Tensor:
        x = self.upsample(x)

        # Handle potential size mismatch from odd-dimension inputs
        if x.shape != skip.shape:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )

        skip = self.attention(g=x, x=skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class AttentionUNet(nn.Module):
    """
    Attention U-Net for binary lung mask segmentation.

    Input: (B, 1, H, W) grayscale image (H=W=256 by default)
    Output: (B, 1, H, W) sigmoid probability map

    Args:
        in_channels: Number of input channels (1 for grayscale CXR).
        out_channels: Number of output channels (1 for binary mask).
        features: List of feature channel counts at each encoder level.
        dropout: Dropout rate for DoubleConv blocks.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: List[int] = (32, 64, 128, 256),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Encoder
        self.encoders = nn.ModuleList()
        in_ch = in_channels
        for feat in features:
            self.encoders.append(EncoderBlock(in_ch, feat, dropout=dropout))
            in_ch = feat

        # Bridge (bottleneck)
        self.bridge = DoubleConv(features[-1], features[-1] * 2, dropout=dropout)

        # Decoder
        self.decoders = nn.ModuleList()
        bridge_channels = features[-1] * 2
        decoder_features = list(reversed(features))

        for i, feat in enumerate(decoder_features):
            skip_ch = feat
            out_ch = feat if i < len(decoder_features) - 1 else feat
            self.decoders.append(
                DecoderBlock(
                    in_channels=bridge_channels if i == 0 else decoder_features[i - 1],
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                    dropout=dropout,
                )
            )
            bridge_channels = feat

        # Output convolution
        self.output_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 1, H, W).

        Returns:
            Output mask logits (B, 1, H, W). Apply sigmoid for probabilities.
        """
        # Encoder path
        skips = []
        for encoder in self.encoders:
            skip, x = encoder(x)
            skips.append(skip)

        # Bridge
        x = self.bridge(x)

        # Decoder path
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        return self.output_conv(x)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_unet(
    in_channels: int = 1,
    out_channels: int = 1,
    features: List[int] = (32, 64, 128, 256),
    dropout: float = 0.1,
) -> AttentionUNet:
    """
    Factory function to build Attention U-Net.

    Args:
        in_channels: Input channels (1 for grayscale).
        out_channels: Output channels (1 for binary segmentation).
        features: Encoder feature channel sizes.
        dropout: Dropout rate.

    Returns:
        Initialized AttentionUNet model.
    """
    model = AttentionUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        features=features,
        dropout=dropout,
    )
    n_params = model.count_parameters()
    logger.info(
        f"AttentionUNet built: "
        f"in_channels={in_channels}, features={features}, "
        f"trainable params={n_params:,}"
    )
    return model
