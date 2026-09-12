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
        return out.float()


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
            states.append(out[:, :2].float())
            full_states.append(out.float())
        out = out.float()
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
        # Demapper activations can exceed float16 range at high-SNR samples.
        # Keep this learned boundary in float32 while retaining AMP elsewhere.
        with torch.autocast(device_type=x.device.type, enabled=False):
            out = x.float()
            for block in self.blocks:
                out = block(out)
        return out.float()


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
        mask = pilot_mask > 0
        frequency_locations = torch.nonzero(mask.any(dim=-1), as_tuple=False)
        time_locations = torch.nonzero(mask.any(dim=-2), as_tuple=False)
        pair_count = n * nt
        if (
            frequency_locations.shape[0] == 0
            or frequency_locations.shape[0] % pair_count
            or time_locations.shape[0] % pair_count
        ):
            raise ValueError("each batch/layer pair must have a uniform pilot grid")
        n_pilot_frequencies = frequency_locations.shape[0] // pair_count
        n_pilot_symbols = time_locations.shape[0] // pair_count
        frequencies = frequency_locations[:, 2].reshape(
            n, nt, n_pilot_frequencies
        )
        symbols = time_locations[:, 2].reshape(n, nt, n_pilot_symbols)

        by_pair = raw.permute(0, 2, 1, 3, 4)
        frequency_indices = frequencies[:, :, None, :, None].expand(
            n, nt, nr, n_pilot_frequencies, s
        )
        compact = torch.gather(by_pair, 3, frequency_indices)
        time_indices = symbols[:, :, None, None, :].expand(
            n, nt, nr, n_pilot_frequencies, n_pilot_symbols
        )
        compact = torch.gather(compact, 4, time_indices)
        denoiser_input = torch.stack((compact.real, compact.imag), dim=3).reshape(
            n * nt * nr, 2, n_pilot_frequencies, n_pilot_symbols
        )
        denoised = self.denoise(denoiser_input).reshape(
            n, nt, nr, 2, n_pilot_frequencies, n_pilot_symbols
        )
        estimate = torch.complex(denoised[:, :, :, 0], denoised[:, :, :, 1])
        flat_indices = (
            frequencies[..., :, None] * s + symbols[..., None, :]
        ).reshape(n, nt, 1, -1).expand(n, nt, nr, -1)
        full = raw.new_zeros(n, nt, nr, f * s)
        full.scatter_(3, flat_indices, estimate.reshape(n, nt, nr, -1))
        return full.reshape(n, nt, nr, f, s).permute(0, 2, 1, 3, 4)

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
        coordinates = self._coordinates(n, f, s, received.device, received.real.dtype)
        detector_input = torch.cat(
            (
                lmmse.real.unsqueeze(2),
                lmmse.imag.unsqueeze(2),
                rzf.real.unsqueeze(2),
                rzf.imag.unsqueeze(2),
                coordinates.unsqueeze(1).expand(-1, layers, -1, -1, -1),
            ),
            dim=2,
        ).reshape(n * layers, 6, f, s)
        detector_features, states, full_states = self.detector(
            detector_input, return_full_states=True
        )
        demapped = self.demapper(detector_features)
        logits = demapped.reshape(n, layers, demapped.shape[1], f, s)
        all_states = [state.reshape(n, layers, 2, f, s) for state in states]
        all_detector_states = [
            state.reshape(n, layers, self.model_config.detector_channels, f, s)
            for state in full_states
        ]
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
