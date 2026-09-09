from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from .config import EqDeepRxConfig


MODULATION_BITS: Dict[str, int] = {"16QAM": 4, "64QAM": 6}


def bits_per_symbol(modulation: str) -> int:
    try:
        return MODULATION_BITS[modulation.upper()]
    except KeyError as exc:
        raise ValueError(f"Unsupported modulation: {modulation}") from exc


def _constellation(modulation: str, *, device: torch.device | str, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, int]:
    bps = bits_per_symbol(modulation)
    levels_per_dim = 2 ** (bps // 2)
    gray = torch.arange(levels_per_dim, device=device) ^ (torch.arange(levels_per_dim, device=device) >> 1)
    levels = 2 * torch.arange(levels_per_dim, device=device, dtype=dtype) - (levels_per_dim - 1)
    pam = torch.zeros(levels_per_dim, device=device, dtype=dtype)
    pam[gray] = levels
    real = pam.repeat_interleave(levels_per_dim)
    imag = pam.repeat(levels_per_dim)
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
    """Max-log LLRs with the positive-logit means bit-one convention."""

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
        llrs[bit] = (d0 - d1) / variance
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
    snr_db: float


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
        delays = torch.as_tensor(self._DELAY_TAPS, device=self.device)
        powers = 10 ** (torch.as_tensor(self._TAP_POWER_DB, device=self.device) / 10.0)
        powers = powers / powers.sum()
        taps = (torch.randn(batch_size, nr, n_layers, len(delays), s, generator=generator) + 1j * torch.randn(batch_size, nr, n_layers, len(delays), s, generator=generator)).to(self.device)
        taps = taps * torch.sqrt(powers.to(self.device).view(1, 1, 1, -1, 1) / 2.0)
        freq = torch.arange(f, device=self.device, dtype=torch.float32).view(1, 1, 1, f, 1)
        phase = -2.0 * math.pi * freq * delays.to(torch.float32).view(1, 1, 1, 1, -1) / float(self.config.n_fft)
        steering = torch.exp(1j * phase)
        return (taps.unsqueeze(3) * steering.unsqueeze(-1)).sum(dim=4)

    def generate_batch(
        self,
        *,
        batch_size: int,
        n_layers: int,
        pilot_count: int,
        snr_db: float,
        seed: int,
        add_interference: bool = False,
    ) -> SignalBatch:
        """Generate an uncoded batch with deterministic channel/noise for `seed`."""

        self.config.validate_layer_count(n_layers)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        bps = bits_per_symbol(self.config.modulation)
        pilot_mask_one = self._pilot_layout(n_layers, pilot_count)
        pilot_mask = pilot_mask_one.unsqueeze(0).expand(batch_size, -1, -1, -1).clone()
        all_pilots = pilot_mask.any(dim=1, keepdim=True)
        data_mask = (~all_pilots.bool()).to(torch.float32)
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
        noise_variance = signal_power / (10.0 ** (float(snr_db) / 10.0))
        noise = (torch.randn(received_waveform.shape, generator=generator) + 1j * torch.randn(received_waveform.shape, generator=generator)).to(self.device)
        received_waveform = received_waveform + noise * torch.sqrt(noise_variance / 2.0)
        if add_interference:
            interference = (torch.randn(received_waveform.shape, generator=generator) + 1j * torch.randn(received_waveform.shape, generator=generator)).to(self.device)
            received_waveform = received_waveform + interference * torch.sqrt(noise_variance * 10.0 / 2.0)
        received = self._demodulate(received_waveform)
        return SignalBatch(
            received=received,
            transmitted=transmitted,
            pilot_symbols=pilot_symbols,
            pilot_mask=pilot_mask,
            data_mask=data_mask,
            target_bits=target_bits,
            true_channel=channel,
            noise_variance=noise_variance.flatten(),
            snr_db=float(snr_db),
        )

