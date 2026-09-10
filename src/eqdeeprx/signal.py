from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from .config import EqDeepRxConfig


MODULATION_BITS: Dict[str, int] = {
    "QPSK": 2,
    "16QAM": 4,
    "64QAM": 6,
    "256QAM": 8,
}


def bits_per_symbol(modulation: str) -> int:
    try:
        return MODULATION_BITS[modulation.upper()]
    except KeyError as exc:
        raise ValueError(f"Unsupported modulation: {modulation}") from exc


def _constellation(modulation: str, *, device: torch.device | str, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, int]:
    bps = bits_per_symbol(modulation)
    indices = torch.arange(2**bps, device=device)
    labels = ((indices[:, None] >> torch.arange(bps - 1, -1, -1, device=device)) & 1).to(dtype)

    def pam_gray(bits: torch.Tensor) -> torch.Tensor:
        value = 1.0 - 2.0 * bits[:, -1]
        for bit_index in range(bits.shape[1] - 2, -1, -1):
            value = (1.0 - 2.0 * bits[:, bit_index]) * (
                float(2 ** (bits.shape[1] - bit_index - 1)) - value
            )
        return value

    real = pam_gray(labels[:, 0::2])
    imag = pam_gray(labels[:, 1::2])
    constellation = torch.complex(real, imag)
    constellation = constellation / torch.sqrt((constellation.abs() ** 2).mean())
    return constellation, bps


def qam_modulate(bits: torch.Tensor, modulation: str = "16QAM") -> torch.Tensor:
    """Map a final-dimension binary tensor to unit-average-power Gray QAM."""

    constellation, bps = _constellation(modulation, device=bits.device)
    if bits.shape[-1] != bps:
        raise ValueError(f"{modulation} requires {bps} bits in the final dimension")
    powers = 2 ** torch.arange(bps - 1, -1, -1, device=bits.device)
    indices = (bits.to(torch.long) * powers).sum(dim=-1)
    return constellation[indices]


def qam_demapper_llr(symbols: torch.Tensor, noise_variance: torch.Tensor | float, modulation: str = "16QAM", max_bits: int = 8) -> torch.Tensor:
    """Max-log LLRs using Eq. (10): positive values mean bit zero."""

    constellation, bps = _constellation(modulation, device=symbols.device, dtype=symbols.real.dtype)
    if max_bits < bps:
        raise ValueError("max_bits must cover the requested modulation")
    labels = ((torch.arange(constellation.numel(), device=symbols.device)[:, None] >> torch.arange(bps - 1, -1, -1, device=symbols.device)) & 1).to(symbols.dtype)
    distances = (symbols.unsqueeze(-1) - constellation.view(*([1] * symbols.dim()), -1)).abs() ** 2
    variance = torch.as_tensor(noise_variance, device=symbols.device, dtype=symbols.real.dtype).clamp_min(1e-8)
    while variance.dim() < symbols.dim():
        variance = variance.unsqueeze(-1)
    llrs = symbols.real.new_zeros((bps,) + symbols.shape)
    for bit in range(bps):
        d0 = distances[..., labels[:, bit] == 0].amin(dim=-1)
        d1 = distances[..., labels[:, bit] == 1].amin(dim=-1)
        llrs[bit] = (d1 - d0) / variance
    if max_bits == bps:
        return llrs
    return torch.cat((llrs, symbols.real.new_zeros((max_bits - bps,) + symbols.shape)), dim=0)


@dataclass(frozen=True)
class SignalBatch:
    received: torch.Tensor
    transmitted: torch.Tensor
    pilot_symbols: torch.Tensor
    pilot_mask: torch.Tensor
    data_mask: torch.Tensor
    target_bits: torch.Tensor
    true_channel: torch.Tensor
    noise_variance: torch.Tensor
    snr_db: float | torch.Tensor
    backend: str = "fast_ofdm"
    interference_present: bool = False
    sinr_db: float | None = None
    realized_sinr_db: float | None = None
    channel_model: str = "fast_tdl"
    speed_mps_range: Tuple[float, float] | None = None
    sample_rate_hz: float | None = None


class OFDMSystem:
    """Small deterministic OFDM link used for paper-aligned smoke/evaluation runs."""

    _DELAY_TAPS = (0, 1, 3, 5)
    _TAP_POWER_DB = (0.0, -2.0, -5.0, -8.0)

    def __init__(self, config: EqDeepRxConfig, device: torch.device | str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        if config.n_fft < config.n_subcarriers:
            raise ValueError("n_fft must be at least n_subcarriers")

    def _pilot_layout(self, n_layers: int, pilot_count: int) -> torch.Tensor:
        self.config.validate_layer_count(n_layers)
        if pilot_count not in (1, 2):
            raise ValueError("pilot_count must be 1 or 2")
        mask = torch.zeros(n_layers, self.config.n_subcarriers, self.config.n_ofdm_symbols, device=self.device)
        symbols = (2, 11)[:pilot_count]
        spacing = self.config.receiver.pilot_spacing
        for layer in range(n_layers):
            positions = torch.arange(layer % spacing, self.config.n_subcarriers, spacing, device=self.device)
            mask[layer, positions[:, None], torch.as_tensor(symbols, device=self.device)[None, :]] = 1.0
        return mask

    def _modulate(self, grid: torch.Tensor) -> torch.Tensor:
        batch, streams, _, symbols = grid.shape
        full = torch.zeros(batch, streams, self.config.n_fft, symbols, dtype=torch.cfloat, device=grid.device)
        start = (self.config.n_fft - self.config.n_subcarriers) // 2
        full[:, :, start : start + self.config.n_subcarriers] = grid
        time_domain = torch.fft.ifft(full, dim=2)
        cp = time_domain[:, :, -self.config.cyclic_prefix :]
        return torch.cat((cp, time_domain), dim=2)

    def _demodulate(self, waveform: torch.Tensor) -> torch.Tensor:
        start = self.config.cyclic_prefix
        time_domain = waveform[:, :, start : start + self.config.n_fft]
        full = torch.fft.fft(time_domain, dim=2)
        first = (self.config.n_fft - self.config.n_subcarriers) // 2
        return full[:, :, first : first + self.config.n_subcarriers]

    def _channel(self, batch_size: int, n_layers: int, generator: torch.Generator) -> torch.Tensor:
        nr, f, s = self.config.n_rx_antennas, self.config.n_subcarriers, self.config.n_ofdm_symbols
        sample_rate = self.config.sample_rate_hz
        max_delay_samples = min(
            float(max(self.config.cyclic_prefix - 1, 0)),
            float(self.config.delay_spread_ns_range[1]) * 1e-9 * sample_rate,
        )
        delay_spread = torch.rand(batch_size, generator=generator) * (
            float(self.config.delay_spread_ns_range[1]) - float(self.config.delay_spread_ns_range[0])
        ) + float(self.config.delay_spread_ns_range[0])
        delay_spread = (delay_spread * 1e-9 * sample_rate).clamp_min(0.0).clamp_max(max_delay_samples)
        relative_delays = torch.as_tensor((0.0, 0.2, 0.5, 1.0), dtype=torch.float32).view(1, 1, 1, 1, -1)
        delays = relative_delays * delay_spread.view(batch_size, 1, 1, 1, 1)
        powers = 10 ** (torch.as_tensor(self._TAP_POWER_DB, device=self.device) / 10.0)
        powers = powers / powers.sum()
        base = (torch.randn(batch_size, nr, n_layers, 4, 1, generator=generator) + 1j * torch.randn(batch_size, nr, n_layers, 4, 1, generator=generator)).to(self.device)
        speed = torch.rand(batch_size, generator=generator) * (
            float(self.config.speed_mps_range[1]) - float(self.config.speed_mps_range[0])
        ) + float(self.config.speed_mps_range[0])
        doppler_max = speed * float(self.config.carrier_frequency_hz) / 299_792_458.0
        doppler = (torch.rand(batch_size, 1, 1, 4, 1, generator=generator) * 2.0 - 1.0).to(self.device) * doppler_max.to(self.device).view(batch_size, 1, 1, 1, 1)
        symbol_time = (float(self.config.n_fft + self.config.cyclic_prefix) / sample_rate)
        symbol_index = torch.arange(s, device=self.device, dtype=torch.float32).view(1, 1, 1, 1, s)
        taps = base * torch.exp(1j * 2.0 * math.pi * doppler * symbol_time * symbol_index)
        taps = taps * torch.sqrt(powers.to(self.device).view(1, 1, 1, 4, 1) / 2.0)
        freq = torch.arange(f, device=self.device, dtype=torch.float32).view(1, 1, 1, f, 1)
        phase = -2.0 * math.pi * freq * delays.to(self.device) / float(self.config.n_fft)
        steering = torch.exp(1j * phase).unsqueeze(-1)
        return (taps.unsqueeze(3) * steering).sum(dim=4)

    def generate_batch(
        self,
        *,
        batch_size: int,
        n_layers: int,
        pilot_count: int,
        snr_db: float | torch.Tensor,
        seed: int,
        add_interference: bool = False,
        return_true_channel: bool = True,
    ) -> SignalBatch:
        """Generate an uncoded batch with deterministic channel/noise for `seed`."""

        self.config.validate_layer_count(n_layers)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        bps = bits_per_symbol(self.config.modulation)
        pilot_mask_one = self._pilot_layout(n_layers, pilot_count)
        pilot_mask = pilot_mask_one.unsqueeze(0).expand(batch_size, -1, -1, -1).clone()
        pilot_ofdm_symbols = pilot_mask.any(dim=(1, 2))
        data_mask = (~pilot_ofdm_symbols[:, None, None, :]).expand(
            -1, 1, self.config.n_subcarriers, -1
        ).to(torch.float32).clone()
        bits = torch.randint(0, 2, (batch_size, n_layers, self.config.n_subcarriers, self.config.n_ofdm_symbols, bps), generator=generator).float().to(self.device)
        data_symbols = qam_modulate(bits, self.config.modulation)
        pilot_signs = torch.randint(0, 2, pilot_mask.shape, generator=generator).float().to(self.device) * 2.0 - 1.0
        pilot_imag = torch.randint(0, 2, pilot_mask.shape, generator=generator).float().to(self.device) * 2.0 - 1.0
        pilot_symbols = (pilot_signs + 1j * pilot_imag) / math.sqrt(2.0) * pilot_mask
        transmitted = data_symbols * data_mask + pilot_symbols
        target_bits = bits.permute(0, 1, 4, 2, 3).contiguous() * data_mask.unsqueeze(1)

        channel = self._channel(batch_size, n_layers, generator)
        full_grid = torch.zeros(batch_size, n_layers, self.config.n_fft, self.config.n_ofdm_symbols, dtype=torch.cfloat, device=self.device)
        start = (self.config.n_fft - self.config.n_subcarriers) // 2
        full_grid[:, :, start : start + self.config.n_subcarriers] = transmitted
        received_full = torch.einsum("bnmfs,bmfs->bnfs", channel, transmitted)
        received_full_fft = torch.zeros(batch_size, self.config.n_rx_antennas, self.config.n_fft, self.config.n_ofdm_symbols, dtype=torch.cfloat, device=self.device)
        received_full_fft[:, :, start : start + self.config.n_subcarriers] = received_full
        received_waveform = torch.fft.ifft(received_full_fft, dim=2)
        cp = received_waveform[:, :, -self.config.cyclic_prefix :]
        received_waveform = torch.cat((cp, received_waveform), dim=2)
        signal_power = received_waveform.abs().square().mean(dim=(1, 2), keepdim=True)
        requested_snr = torch.as_tensor(
            snr_db, dtype=signal_power.dtype, device=self.device
        ).flatten()
        if requested_snr.numel() == 1:
            requested_snr = requested_snr.expand(batch_size)
        if requested_snr.numel() != batch_size:
            raise ValueError("snr_db must be scalar or have one value per sample")
        noise_variance = signal_power / torch.pow(
            10.0, requested_snr.view(batch_size, 1, 1) / 10.0
        )
        noise = (torch.randn(received_waveform.shape, generator=generator) + 1j * torch.randn(received_waveform.shape, generator=generator)).to(self.device)
        received_waveform = received_waveform + noise * torch.sqrt(noise_variance / 2.0)
        if add_interference:
            interference_bits = torch.randint(0, 2, (batch_size, 1, self.config.n_subcarriers, self.config.n_ofdm_symbols, bps), generator=generator).float().to(self.device)
            interference_symbols = qam_modulate(interference_bits, self.config.modulation)
            interference_channel = self._channel(batch_size, 1, generator)
            interference_grid = torch.einsum("bnmfs,bmfs->bnfs", interference_channel, interference_symbols)
            interference_full = torch.zeros(batch_size, self.config.n_rx_antennas, self.config.n_fft, self.config.n_ofdm_symbols, dtype=torch.cfloat, device=self.device)
            interference_full[:, :, start : start + self.config.n_subcarriers] = interference_grid
            interference_waveform = torch.fft.ifft(interference_full, dim=2)
            interference_waveform = torch.cat((interference_waveform[:, :, -self.config.cyclic_prefix :], interference_waveform), dim=2)
            timing_offsets = torch.randint(0, max(1, self.config.cyclic_prefix), (batch_size,), generator=generator)
            for index, offset in enumerate(timing_offsets.tolist()):
                interference_waveform[index] = torch.roll(interference_waveform[index], shifts=int(offset), dims=1)
            inr_db = 10.0 + 5.0 * torch.randn(batch_size, generator=generator)
            target_power = noise_variance * (10.0 ** (inr_db.to(self.device).view(batch_size, 1, 1) / 10.0))
            actual_power = interference_waveform.abs().square().mean(dim=(1, 2), keepdim=True).clamp_min(1e-12)
            interference_waveform = interference_waveform * torch.sqrt(target_power / actual_power)
            received_waveform = received_waveform + interference_waveform
        received = self._demodulate(received_waveform)
        reported_snr: float | torch.Tensor
        if batch_size == 1:
            reported_snr = float(requested_snr.item())
        else:
            reported_snr = requested_snr.detach()
        return SignalBatch(
            received=received,
            transmitted=transmitted,
            pilot_symbols=pilot_symbols,
            pilot_mask=pilot_mask,
            data_mask=data_mask,
            target_bits=target_bits,
            true_channel=(
                channel
                if return_true_channel
                else torch.empty(0, dtype=channel.dtype, device=self.device)
            ),
            noise_variance=noise_variance.flatten(),
            snr_db=reported_snr,
            interference_present=bool(add_interference),
            speed_mps_range=self.config.speed_mps_range,
            sample_rate_hz=self.config.sample_rate_hz,
        )
