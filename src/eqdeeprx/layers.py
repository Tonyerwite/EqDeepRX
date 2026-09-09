from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int]):
        super().__init__()
        padding = tuple((size - 1) // 2 for size in kernel_size)
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class SubsampledResidualBlock(nn.Module):
    """EqDeepRx Eq. (11)-(12) with nearest-neighbor frequency subsampling."""

    def __init__(self, in_channels: int, out_channels: int, *, downsample: int = 1, frequency_only: bool = False):
        super().__init__()
        if downsample < 1 or downsample & (downsample - 1):
            raise ValueError("downsample must be a positive power of two")
        first_kernel = (13, 1) if frequency_only else (13, 1)
        second_kernel = (13, 1) if frequency_only else (1, 13)
        self.downsample = int(downsample)
        self.frequency_only = frequency_only
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, first_kernel)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, second_kernel)
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.projection(F.relu(self.bn1(x))) if not isinstance(self.projection, nn.Identity) else x
        residual = F.relu(self.bn1(x))
        original_size = residual.shape[-2:]
        if self.downsample > 1:
            residual = F.interpolate(residual, size=(max(1, original_size[0] // self.downsample), original_size[1]), mode="nearest")
        residual = self.conv1(residual)
        residual = self.conv2(F.relu(self.bn2(residual)))
        if residual.shape[-2:] != original_size:
            residual = F.interpolate(residual, size=original_size, mode="nearest")
        return residual + shortcut


class TimeMixer(nn.Module):
    """Lightweight shared mixer over pilot OFDM-symbol positions."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, f, s = x.shape
        mixed = x.permute(0, 1, 2, 3).reshape(n * c * f, 1, s)
        mixed = self.conv(mixed).reshape(n, c, f, s)
        return x + mixed


class PointwiseResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return out + shortcut

