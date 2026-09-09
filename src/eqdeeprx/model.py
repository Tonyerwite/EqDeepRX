from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn

from .config import EqDeepRxConfig, ModelConfig
from .layers import PointwiseResidualBlock, SubsampledResidualBlock, TimeMixer
from .receiver import estimate_incm, estimate_raw_channel, interpolate_channel, lmmse_equalize, rzf_equalize


class DenoiseNN(nn.Module):
    def __init__(self, *, widths: Tuple[int, ...] = (64, 64, 64, 2), subsamples: Tuple[int, ...] = (1, 4, 2, 1)):
        super().__init__()
        if len(widths) != len(subsamples):
            raise ValueError("widths and subsamples must have equal length")
        blocks = []
        in_channels = 2
        for width, downsample in zip(widths, subsamples):
            blocks.append(SubsampledResidualBlock(in_channels, width, downsample=downsample, frequency_only=True))
            in_channels = width
        self.blocks = nn.ModuleList(blocks)
        self.mixers = nn.ModuleList(TimeMixer() for _ in widths)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for block, mixer in zip(self.blocks, self.mixers):
            out = mixer(block(out))
        return out


class DetectorNN(nn.Module):
    def __init__(self, *, channels: int = 64, sections: int = 4, max_bits: int = 8, detector_subsample: int = 8):
        super().__init__()
        self.channels = channels
        self.sections = sections
        self.project = nn.Conv2d(6, channels, kernel_size=1)
        self.section_blocks = nn.ModuleList()
        for _ in range(sections):
            self.section_blocks.append(
                nn.ModuleList(
                    [
                        SubsampledResidualBlock(channels, channels, downsample=1, frequency_only=False),
                        SubsampledResidualBlock(channels, channels, downsample=detector_subsample, frequency_only=False),
                    ]
                )
            )
        self.max_bits = max_bits

    def forward(self, x: torch.Tensor, *, return_full_states: bool = False):
        out = self.project(x)
        states: List[torch.Tensor] = []
        full_states: List[torch.Tensor] = []
        for blocks in self.section_blocks:
            updated = blocks[1](blocks[0](out))
            out = out + updated
            states.append(out[:, :2])
            full_states.append(out)
        if return_full_states:
            return out, states, full_states
        return out, states


class DemapperNN(nn.Module):
    def __init__(self, *, in_channels: int = 64, widths: Tuple[int, ...] = (32, 32, 32, 8)):
        super().__init__()
        blocks = []
        current = in_channels
        for width in widths:
            blocks.append(PointwiseResidualBlock(current, width))
            current = width
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for block in self.blocks:
            out = block(out)
        return out


class EqDeepRx(nn.Module):
    """EqDeepRx hybrid receiver returning uncoded bit logits."""

    def __init__(self, config: EqDeepRxConfig | None = None, *, model_config: ModelConfig | None = None):
        super().__init__()
        self.config = config or EqDeepRxConfig()
        self.model_config = model_config or self.config.model
        self.n_rx = self.config.n_rx_antennas
        self.denoise = DenoiseNN(widths=self.model_config.denoise_widths, subsamples=self.model_config.denoise_subsamples)
        self.detector = DetectorNN(
            channels=self.model_config.detector_channels,
            sections=self.model_config.detector_sections,
            max_bits=self.model_config.max_bits,
            detector_subsample=self.model_config.detector_subsample,
        )
        self.demapper = DemapperNN(in_channels=self.model_config.detector_channels, widths=self.model_config.demapper_widths)

    def _denoise_channel(self, raw: torch.Tensor, pilot_mask: torch.Tensor) -> torch.Tensor:
        n, nr, nt, f, s = raw.shape
        features = torch.stack((raw.real, raw.imag), dim=2).reshape(n * nr * nt, 2, f, s)
        denoised = self.denoise(features).reshape(n, nr, nt, 2, f, s)
        complex_estimate = torch.complex(denoised[:, :, :, 0], denoised[:, :, :, 1])
        return complex_estimate * pilot_mask[:, None]

    @staticmethod
    def _coordinates(batch: int, f: int, s: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        freq = torch.linspace(-1.0, 1.0, f, device=device, dtype=dtype).view(1, 1, f, 1).expand(batch, 1, f, s)
        time = torch.linspace(-1.0, 1.0, s, device=device, dtype=dtype).view(1, 1, 1, s).expand(batch, 1, f, s)
        return torch.cat((freq, time), dim=1)

    def forward(self, received: torch.Tensor, pilot_symbols: torch.Tensor, pilot_mask: torch.Tensor, *, return_aux: bool = False):
        raw = estimate_raw_channel(received, pilot_symbols, pilot_mask)
        denoised = self._denoise_channel(raw, pilot_mask)
        channel = interpolate_channel(denoised, pilot_mask)
        covariance = estimate_incm(received, channel, pilot_symbols, pilot_mask, coherence_bandwidth=self.config.receiver.incm_coherence_bandwidth)
        lmmse = lmmse_equalize(received, channel, covariance, coherence_bandwidth=self.config.receiver.incm_coherence_bandwidth)
        rzf = rzf_equalize(received, channel, alpha=self.config.receiver.rzf_alpha)
        n, layers, f, s = lmmse.shape
        logits_per_layer = []
        all_states: List[torch.Tensor] = []
        all_detector_states: List[torch.Tensor] = []
        coordinates = self._coordinates(n, f, s, received.device, received.real.dtype)
        for layer in range(layers):
            detector_input = torch.cat((lmmse[:, layer].real.unsqueeze(1), lmmse[:, layer].imag.unsqueeze(1), rzf[:, layer].real.unsqueeze(1), rzf[:, layer].imag.unsqueeze(1), coordinates), dim=1)
            detector_features, states, full_states = self.detector(detector_input, return_full_states=True)
            logits_per_layer.append(self.demapper(detector_features))
            if not all_states:
                all_states = [state.unsqueeze(1) for state in states]
            else:
                all_states = [torch.cat((old, state.unsqueeze(1)), dim=1) for old, state in zip(all_states, states)]
            if not all_detector_states:
                all_detector_states = [state.unsqueeze(1) for state in full_states]
            else:
                all_detector_states = [torch.cat((old, state.unsqueeze(1)), dim=1) for old, state in zip(all_detector_states, full_states)]
        logits = torch.stack(logits_per_layer, dim=1)
        if layers == 1:
            logits = logits[:, 0]
        if not return_aux:
            return logits
        return logits, {
            "lmmse": lmmse,
            "rzf": rzf,
            "channel": channel,
            "covariance": covariance,
            "symbol_states": all_states,
            "detector_states": all_detector_states,
        }

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
