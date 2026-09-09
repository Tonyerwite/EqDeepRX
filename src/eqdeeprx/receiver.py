from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class EqualizerOutput:
    lmmse: torch.Tensor
    rzf: torch.Tensor
    covariance: torch.Tensor


def estimate_raw_channel(received: torch.Tensor, pilot_symbols: torch.Tensor, pilot_mask: torch.Tensor) -> torch.Tensor:
    """Equation (8): rank-one LS channel estimates at orthogonal DMRS REs."""

    if received.dim() != 4 or pilot_symbols.dim() != 4:
        raise ValueError("received and pilot_symbols must be [N, antenna/layer, F, S]")
    pilot_energy = pilot_symbols.abs().square().clamp_min(1e-8)
    raw = received[:, :, None] * pilot_symbols[:, None].conj() / pilot_energy[:, None]
    return raw * pilot_mask[:, None]


def _interp_1d(values: torch.Tensor, known: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    if known.numel() == 0:
        return torch.zeros_like(query, dtype=values.dtype)
    if known.numel() == 1:
        return values[0].expand(query.shape)
    right = torch.searchsorted(known, query).clamp(1, known.numel() - 1)
    left = right - 1
    x0, x1 = known[left], known[right]
    weight = ((query - x0) / (x1 - x0).clamp_min(1e-8)).to(values.real.dtype)
    return values[left] + weight.to(values.dtype) * (values[right] - values[left])


def interpolate_channel(raw: torch.Tensor, pilot_mask: torch.Tensor) -> torch.Tensor:
    """Linear frequency/time interpolation from the sparse pilot grid."""

    n, nr, nt, f, s = raw.shape
    result = torch.zeros_like(raw)
    freq_query = torch.arange(f, device=raw.device, dtype=torch.float32)
    time_query = torch.arange(s, device=raw.device, dtype=torch.float32)
    for batch in range(n):
        for rx in range(nr):
            for layer in range(nt):
                frequency = torch.zeros(s, f, dtype=raw.dtype, device=raw.device)
                time_positions = []
                for symbol in range(s):
                    positions = torch.nonzero(pilot_mask[batch, layer, :, symbol] > 0, as_tuple=False).flatten()
                    if positions.numel():
                        time_positions.append(symbol)
                        frequency[symbol] = _interp_1d(raw[batch, rx, layer, positions, symbol], positions.to(torch.float32), freq_query)
                if not time_positions:
                    continue
                known_t = torch.as_tensor(time_positions, dtype=torch.float32, device=raw.device)
                for carrier in range(f):
                    result[batch, rx, layer, carrier] = _interp_1d(frequency[known_t.long(), carrier], known_t, time_query)
    return result


def estimate_incm(
    received: torch.Tensor,
    channel: torch.Tensor,
    transmitted: torch.Tensor,
    pilot_mask: torch.Tensor,
    *,
    coherence_bandwidth: int = 24,
) -> torch.Tensor:
    """Estimate a shrunken interference-plus-noise covariance per frequency band."""

    n, nr, _, f, s = channel.shape
    n_bands = (f + coherence_bandwidth - 1) // coherence_bandwidth
    covariance = torch.zeros(n, nr, nr, n_bands, dtype=received.dtype, device=received.device)
    for batch in range(n):
        for band in range(n_bands):
            start = band * coherence_bandwidth
            stop = min(f, start + coherence_bandwidth)
            mask = pilot_mask[batch, :, start:stop] > 0
            positions = torch.nonzero(mask, as_tuple=False)
            if positions.numel() == 0:
                covariance[batch, :, :, band] = torch.eye(nr, dtype=received.dtype, device=received.device) * 1e-3
                continue
            residuals = []
            for layer, carrier, symbol in positions.tolist():
                predicted = channel[batch, :, layer, start + carrier, symbol] * transmitted[batch, layer, start + carrier, symbol]
                residuals.append(received[batch, :, start + carrier, symbol] - predicted)
            samples = torch.stack(residuals, dim=0)
            sample_covariance = samples.conj().transpose(0, 1) @ samples / max(samples.shape[0], 1)
            sample_covariance = (sample_covariance + sample_covariance.conj().transpose(-1, -2)) / 2.0
            trace = sample_covariance.diagonal().real.mean().clamp_min(1e-8)
            shrinkage = min(1.0, float(nr) / float(max(samples.shape[0], 1)))
            covariance[batch, :, :, band] = (1.0 - shrinkage) * sample_covariance + shrinkage * trace * torch.eye(nr, dtype=received.dtype, device=received.device)
    return covariance


def _solve_equalizer(channel: torch.Tensor, received: torch.Tensor, covariance: torch.Tensor | None, alpha: float, coherence_bandwidth: int = 24) -> torch.Tensor:
    n, nr, nt, f, s = channel.shape
    h = channel.permute(0, 3, 4, 1, 2)
    y = received.permute(0, 2, 3, 1).unsqueeze(-1)
    eye_t = torch.eye(nt, dtype=channel.dtype, device=channel.device).view(1, 1, 1, nt, nt)
    if covariance is None:
        gram = h.conj().transpose(-1, -2) @ h + alpha * eye_t
        weights = torch.linalg.solve(gram, h.conj().transpose(-1, -2))
    else:
        bands = covariance.shape[-1]
        band_indices = torch.div(torch.arange(f, device=channel.device), max(1, coherence_bandwidth), rounding_mode="floor").clamp_max(bands - 1)
        r = covariance.permute(0, 3, 1, 2)[:, band_indices]
        hh = h @ h.conj().transpose(-1, -2) + r.unsqueeze(2)
        weights = h.conj().transpose(-1, -2) @ torch.linalg.inv(hh)
    raw = (weights @ y).squeeze(-1)
    gain = torch.diagonal(weights @ h, dim1=-2, dim2=-1)
    gain = torch.where(gain.abs() < 1e-8, torch.ones_like(gain), gain)
    equalized = raw / gain
    return equalized.permute(0, 3, 1, 2).contiguous()


def rzf_equalize(received: torch.Tensor, channel: torch.Tensor, *, alpha: float = 1e-4) -> torch.Tensor:
    return _solve_equalizer(channel, received, None, alpha)


def lmmse_equalize(received: torch.Tensor, channel: torch.Tensor, covariance: torch.Tensor, *, coherence_bandwidth: int = 24) -> torch.Tensor:
    return _solve_equalizer(channel, received, covariance, 0.0, coherence_bandwidth)


def equalize_parallel(
    received: torch.Tensor,
    raw_channel: torch.Tensor,
    transmitted: torch.Tensor,
    pilot_mask: torch.Tensor,
    *,
    alpha: float = 1e-4,
    coherence_bandwidth: int = 24,
) -> EqualizerOutput:
    channel = interpolate_channel(raw_channel, pilot_mask)
    covariance = estimate_incm(received, channel, transmitted, pilot_mask, coherence_bandwidth=coherence_bandwidth)
    return EqualizerOutput(
        lmmse=lmmse_equalize(received, channel, covariance, coherence_bandwidth=coherence_bandwidth),
        rzf=rzf_equalize(received, channel, alpha=alpha),
        covariance=covariance,
    )
