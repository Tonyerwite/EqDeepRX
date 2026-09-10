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


def _interp_last_dimension(
    values: torch.Tensor, known: torch.Tensor, query: torch.Tensor
) -> torch.Tensor:
    """Linearly interpolate tensors whose sample axis is the final dimension."""

    if known.numel() == 0:
        return values.new_zeros(values.shape[:-1] + query.shape)
    if known.numel() == 1:
        return values[..., :1].expand(values.shape[:-1] + query.shape)
    right = torch.searchsorted(known, query).clamp(1, known.numel() - 1)
    left = right - 1
    x0, x1 = known[left], known[right]
    weight = ((query - x0) / (x1 - x0).clamp_min(1e-8)).to(values.real.dtype)
    return values[..., left] + weight.to(values.dtype) * (
        values[..., right] - values[..., left]
    )


def _interp_1d(values: torch.Tensor, known: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    return _interp_last_dimension(values, known, query)


def interpolate_channel(raw: torch.Tensor, pilot_mask: torch.Tensor) -> torch.Tensor:
    """Linear frequency/time interpolation from the sparse pilot grid."""

    n, nr, nt, f, s = raw.shape
    mask = pilot_mask > 0
    frequency_locations = torch.nonzero(mask.any(dim=-1), as_tuple=False)
    time_locations = torch.nonzero(mask.any(dim=-2), as_tuple=False)
    pair_count = n * nt
    if frequency_locations.shape[0] == 0 or time_locations.shape[0] == 0:
        return torch.zeros_like(raw)
    if (
        frequency_locations.shape[0] % pair_count
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

    frequency_values = compact.permute(0, 1, 2, 4, 3)
    if n_pilot_frequencies == 1:
        frequency_values = frequency_values.expand(-1, -1, -1, -1, f)
    else:
        known_frequencies = frequencies.to(torch.float32).contiguous()
        frequency_query = torch.arange(
            f, device=raw.device, dtype=torch.float32
        ).view(1, 1, f).expand(n, nt, -1)
        right = torch.searchsorted(
            known_frequencies, frequency_query.contiguous()
        ).clamp(1, n_pilot_frequencies - 1)
        left = right - 1
        gather_shape = (n, nt, nr, n_pilot_symbols, f)
        left_values = torch.gather(
            frequency_values,
            -1,
            left[:, :, None, None].expand(gather_shape),
        )
        right_values = torch.gather(
            frequency_values,
            -1,
            right[:, :, None, None].expand(gather_shape),
        )
        x0 = torch.gather(known_frequencies, -1, left)
        x1 = torch.gather(known_frequencies, -1, right)
        weight = (frequency_query - x0) / (x1 - x0).clamp_min(1e-8)
        frequency_values = left_values + weight[:, :, None, None].to(
            raw.dtype
        ) * (right_values - left_values)
    frequency_values = frequency_values.permute(0, 1, 2, 4, 3)

    if n_pilot_symbols == 1:
        interpolated = frequency_values.expand(-1, -1, -1, -1, s)
    else:
        known_symbols = symbols.to(torch.float32).contiguous()
        time_query = torch.arange(
            s, device=raw.device, dtype=torch.float32
        ).view(1, 1, s).expand(n, nt, -1)
        right = torch.searchsorted(
            known_symbols, time_query.contiguous()
        ).clamp(1, n_pilot_symbols - 1)
        left = right - 1
        gather_shape = (n, nt, nr, f, s)
        left_values = torch.gather(
            frequency_values,
            -1,
            left[:, :, None, None].expand(gather_shape),
        )
        right_values = torch.gather(
            frequency_values,
            -1,
            right[:, :, None, None].expand(gather_shape),
        )
        x0 = torch.gather(known_symbols, -1, left)
        x1 = torch.gather(known_symbols, -1, right)
        weight = (time_query - x0) / (x1 - x0).clamp_min(1e-8)
        interpolated = left_values + weight[:, :, None, None].to(
            raw.dtype
        ) * (right_values - left_values)
    return interpolated.permute(0, 2, 1, 3, 4).contiguous()


def smooth_pilot_channel_frequency(
    raw: torch.Tensor,
    pilot_mask: torch.Tensor,
    *,
    window: int = 3,
) -> torch.Tensor:
    """Apply the conventional fixed frequency smoother on sparse pilot grids."""

    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    n, nr, nt, _, n_symbols = raw.shape
    smoothed = torch.zeros_like(raw)
    radius = window // 2
    for batch in range(n):
        for layer in range(nt):
            for symbol in range(n_symbols):
                positions = torch.nonzero(
                    pilot_mask[batch, layer, :, symbol] > 0,
                    as_tuple=False,
                ).flatten()
                for pilot_index, carrier in enumerate(positions.tolist()):
                    start = max(0, pilot_index - radius)
                    stop = min(positions.numel(), pilot_index + radius + 1)
                    smoothed[batch, :, layer, carrier, symbol] = raw[
                        batch, :, layer, positions[start:stop], symbol
                    ].mean(dim=-1)
    return smoothed


def oas_complex_shrinkage(
    sample_covariance: torch.Tensor, n_samples: int | torch.Tensor
) -> torch.Tensor:
    """Return the complex-Gaussian OAS shrinkage coefficient.

    For a Hermitian SCM ``S`` with dimension ``p`` and ``n`` residual
    samples, the complex OAS closed form used in the paper's complex-OAS
    reference is

    ``rho = min(1, (p - gamma/p) / ((n - 1/p) * (gamma - 1)))``

    with ``gamma = p*tr(S^2)/tr(S)^2``.  The coefficient is clamped to
    ``[0,1]`` and degenerate zero/one-dimensional cases are handled without
    introducing NaNs.
    """

    if sample_covariance.ndim < 2 or sample_covariance.shape[-1] != sample_covariance.shape[-2]:
        raise ValueError("sample_covariance must have square trailing dimensions")
    if not isinstance(n_samples, torch.Tensor) and n_samples < 1:
        raise ValueError("n_samples must be positive")
    p = sample_covariance.shape[-1]
    if p <= 1:
        return sample_covariance.real.new_zeros(sample_covariance.shape[:-2])
    hermitian = (sample_covariance + sample_covariance.conj().transpose(-1, -2)) / 2.0
    trace = hermitian.diagonal(dim1=-2, dim2=-1).real.sum(dim=-1)
    trace_sq = trace.square()
    tr_s2 = torch.diagonal(hermitian @ hermitian, dim1=-2, dim2=-1).real.sum(dim=-1).clamp_min(0.0)
    gamma = (float(p) * tr_s2 / trace_sq.clamp_min(torch.finfo(trace.dtype).eps)).clamp_min(1.0)
    numerator = float(p) - gamma / float(p)
    sample_count = torch.as_tensor(
        n_samples, dtype=trace.dtype, device=trace.device
    ).clamp_min(1.0)
    denominator = (sample_count - 1.0 / float(p)) * (gamma - 1.0)
    coefficient = torch.where(denominator > torch.finfo(trace.dtype).eps, numerator / denominator, torch.ones_like(gamma))
    coefficient = torch.where(trace_sq <= torch.finfo(trace.dtype).eps, torch.zeros_like(coefficient), coefficient)
    return coefficient.clamp(0.0, 1.0)


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
    identity = torch.eye(nr, dtype=received.dtype, device=received.device)
    covariance_by_band = []
    for band in range(n_bands):
        start = band * coherence_bandwidth
        stop = min(f, start + coherence_bandwidth)
        active = (pilot_mask[:, :, start:stop] > 0).any(dim=1)
        sample_count = active.sum(dim=(1, 2))
        predicted = (
            channel[:, :, :, start:stop]
            * transmitted[:, None, :, start:stop]
        ).sum(dim=2)
        residuals = (
            received[:, :, start:stop] - predicted
        ) * active[:, None]
        residuals = residuals.flatten(start_dim=2)
        sample_covariance = (
            residuals @ residuals.conj().transpose(-1, -2)
        ) / sample_count.clamp_min(1).to(received.real.dtype)[:, None, None]
        sample_covariance = (
            sample_covariance
            + sample_covariance.conj().transpose(-1, -2)
        ) / 2.0
        trace = (
            sample_covariance.diagonal(dim1=-2, dim2=-1)
            .real.sum(dim=-1)
            .clamp_min(1e-8)
        )
        shrinkage = oas_complex_shrinkage(sample_covariance, sample_count)
        target = (trace / float(nr))[:, None, None] * identity
        shrunken = (
            (1.0 - shrinkage)[:, None, None] * sample_covariance
            + shrinkage[:, None, None] * target
        )
        shrunken = torch.where(
            (sample_count > 0)[:, None, None],
            shrunken,
            identity * 1e-3,
        )
        covariance_by_band.append(shrunken)
    return torch.stack(covariance_by_band, dim=-1)


def _solve_equalizer(
    channel: torch.Tensor,
    received: torch.Tensor,
    covariance: torch.Tensor | None,
    alpha: float,
    coherence_bandwidth: int = 24,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    n, nr, nt, f, s = channel.shape
    h = channel.permute(0, 3, 4, 1, 2)
    y = received.permute(0, 2, 3, 1).unsqueeze(-1)
    eye_t = torch.eye(nt, dtype=channel.dtype, device=channel.device).view(1, 1, 1, nt, nt)
    if covariance is None:
        gram = h.conj().transpose(-1, -2) @ h + alpha * eye_t
        weights = torch.linalg.solve(gram, h.conj().transpose(-1, -2))
        disturbance_variance = None
    else:
        bands = covariance.shape[-1]
        band_indices = torch.div(torch.arange(f, device=channel.device), max(1, coherence_bandwidth), rounding_mode="floor").clamp_max(bands - 1)
        r_by_band = covariance.permute(0, 3, 1, 2)
        eye_r = torch.eye(
            nr, dtype=channel.dtype, device=channel.device
        ).expand(n, bands, nr, nr)
        r_inverse = torch.linalg.solve(r_by_band, eye_r)
        r_inverse = r_inverse[:, band_indices]
        h_hermitian_r_inverse = (
            h.conj().transpose(-1, -2) @ r_inverse.unsqueeze(2)
        )
        gram = h_hermitian_r_inverse @ h + eye_t
        weights = torch.linalg.solve(gram, h_hermitian_r_inverse)
        r = r_by_band[:, band_indices]
        post_covariance = (
            weights
            @ r.unsqueeze(2)
            @ weights.conj().transpose(-1, -2)
        )
        disturbance_variance = torch.diagonal(
            post_covariance, dim1=-2, dim2=-1
        ).real
    raw = (weights @ y).squeeze(-1)
    gain = torch.diagonal(weights @ h, dim1=-2, dim2=-1)
    gain = torch.where(gain.abs() < 1e-8, torch.ones_like(gain), gain)
    equalized = raw / gain
    equalized = equalized.permute(0, 3, 1, 2).contiguous()
    if disturbance_variance is not None:
        disturbance_variance = disturbance_variance / gain.abs().square().clamp_min(
            1e-12
        )
        disturbance_variance = disturbance_variance.permute(0, 3, 1, 2).contiguous()
    return equalized, disturbance_variance


def rzf_equalize(received: torch.Tensor, channel: torch.Tensor, *, alpha: float = 1e-4) -> torch.Tensor:
    return _solve_equalizer(channel, received, None, alpha)[0]


def lmmse_equalize(received: torch.Tensor, channel: torch.Tensor, covariance: torch.Tensor, *, coherence_bandwidth: int = 24) -> torch.Tensor:
    return _solve_equalizer(
        channel, received, covariance, 0.0, coherence_bandwidth
    )[0]


def lmmse_equalize_with_variance(
    received: torch.Tensor,
    channel: torch.Tensor,
    covariance: torch.Tensor,
    *,
    coherence_bandwidth: int = 24,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return unit-gain LMMSE symbols and per-layer disturbance variance."""

    equalized, variance = _solve_equalizer(
        channel, received, covariance, 0.0, coherence_bandwidth
    )
    if variance is None:
        raise RuntimeError("LMMSE disturbance variance was not computed")
    return equalized, variance


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
