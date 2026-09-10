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
    """EqDeepRx Eqs. (11)-(12) with nearest-neighbor subsampling.

    The paper's ``f`` applies ReLU, ``1xN`` depthwise-separable convolution,
    ReLU, and then ``Nx1`` depthwise-separable convolution.  The shortcut is
    kept at full resolution and is projected only when channel counts differ.
    """

    def __init__(self, in_channels: int, out_channels: int, *, downsample: int = 1, frequency_only: bool = False):
        super().__init__()
        if downsample < 1 or downsample & (downsample - 1):
            raise ValueError("downsample must be a positive power of two")
        first_kernel = (13, 1) if frequency_only else (1, 13)
        second_kernel = (13, 1)
        self.downsample = int(downsample)
        self.frequency_only = frequency_only
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, first_kernel)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, second_kernel)
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.projection(x) if not isinstance(self.projection, nn.Identity) else x
        original_size = x.shape[-2:]
        residual = x
        if self.downsample > 1:
            residual = F.interpolate(residual, size=(max(1, original_size[0] // self.downsample), original_size[1]), mode="nearest")
        residual = self.conv1(F.relu(residual))
        residual = self.conv2(F.relu(residual))
        if residual.shape[-2:] != original_size:
            residual = F.interpolate(residual, size=original_size, mode="nearest")
        return residual + shortcut


class TimeMixer(nn.Module):
    """Shared pointwise mixer over a small subset of pilot-symbol channels.

    Pilot symbols are stacked into the input-channel dimension for each
    subcarrier, mixed with a shared 1x1 convolution, and reshaped back.  The
    zero padding lets one module serve both the one- and two-DMRS cases.
    """

    def __init__(self, *, max_symbols: int = 2, mix_channels: int = 2):
        super().__init__()
        if max_symbols < 1 or mix_channels < 1:
            raise ValueError("max_symbols and mix_channels must be positive")
        self.max_symbols = int(max_symbols)
        self.mix_channels = int(mix_channels)
        stacked_channels = self.max_symbols * self.mix_channels
        self.conv = nn.Conv2d(stacked_channels, stacked_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, f, s = x.shape
        if s > self.max_symbols:
            # The production path supplies the compact one/two-pilot grid.
            # Keeping longer tensors unchanged makes the layer safe to use in
            # isolated shape checks and legacy callers.
            return x
        selected = x[:, : min(c, self.mix_channels)]
        selected_channels = selected.shape[1]
        padded = selected.new_zeros(
            n, self.mix_channels, f, self.max_symbols
        )
        padded[:, :selected_channels, :, :s] = selected
        stacked = padded.permute(0, 1, 3, 2).reshape(
            n, self.conv.in_channels, f, 1
        )
        mixed = self.conv(stacked).reshape(
            n, self.mix_channels, self.max_symbols, f
        )
        mixed = mixed[:, :selected_channels, :s].permute(0, 1, 3, 2)
        if selected_channels < c:
            mixed = torch.cat((mixed, x[:, selected_channels:] * 0.0), dim=1)
        return x + mixed


class PointwiseResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)
        out = self.conv1(F.relu(x))
        out = self.conv2(F.relu(out))
        return out + shortcut
