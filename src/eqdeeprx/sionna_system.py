from __future__ import annotations

import math
from typing import Tuple

import torch

from .config import EqDeepRxConfig
from .signal import SignalBatch, bits_per_symbol, qam_modulate


def _cir_to_time_channel_efficient(
    bandwidth: float,
    a: torch.Tensor,
    tau: torch.Tensor,
    l_min: int,
    l_max: int,
    *,
    normalize: bool = False,
) -> torch.Tensor:
    """Sionna-equivalent CIR discretization without a path/time/tap product."""

    if tau.dim() == 4:
        tau = tau.unsqueeze(2).unsqueeze(4)
        tau = tau.expand(-1, -1, -1, -1, a.shape[4], -1)
    lags = torch.arange(
        l_min, l_max + 1, dtype=tau.dtype, device=tau.device
    )
    pulse = torch.sinc(lags - tau.unsqueeze(-1) * float(bandwidth))
    pulse = torch.complex(pulse, torch.zeros_like(pulse))
    taps = torch.einsum("...pt,...pl->...tl", a, pulse)
    if normalize:
        energy = taps.abs().square().sum(dim=6, keepdim=True).mean(
            dim=(2, 4, 5), keepdim=True
        )
        scale = torch.complex(energy.sqrt(), torch.zeros_like(energy))
        taps = torch.where(scale.abs() > 0, taps / scale, taps)
    return taps


def _apply_cir_time_channel_efficient(
    bandwidth: float,
    signals: torch.Tensor,
    a: torch.Tensor,
    tau: torch.Tensor,
    l_min: int,
    l_max: int,
    *,
    normalize: bool = False,
    time_chunk_size: int = 256,
) -> torch.Tensor:
    """Apply a Sionna CIR without materializing the full time/tap response."""

    if signals.ndim != 5:
        raise ValueError(
            "signals must be [batch, component, tx, tx_ant, time]"
        )
    if time_chunk_size < 1:
        raise ValueError("time_chunk_size must be positive")
    if tau.dim() == 4:
        tau = tau.unsqueeze(2).unsqueeze(4)
        tau = tau.expand(-1, -1, -1, -1, a.shape[4], -1)
    lags = torch.arange(
        l_min, l_max + 1, dtype=tau.dtype, device=tau.device
    )
    pulse = torch.sinc(lags - tau.unsqueeze(-1) * float(bandwidth))
    pulse = torch.complex(pulse, torch.zeros_like(pulse))
    n_output_samples = a.shape[-1]
    l_tot = l_max - l_min + 1
    if n_output_samples != signals.shape[-1] + l_tot - 1:
        raise ValueError("CIR and signal time dimensions are inconsistent")

    time = torch.arange(n_output_samples, device=signals.device)[:, None]
    lag = torch.arange(l_tot, device=signals.device)[None, :]
    indices = time - lag
    indices = torch.where(
        (indices >= 0) & (indices < signals.shape[-1]),
        indices,
        signals.shape[-1],
    )
    padded = torch.nn.functional.pad(signals, (0, 1))
    gathered = padded[..., indices].unsqueeze(2).unsqueeze(2)
    output = a.new_zeros(
        a.shape[0],
        signals.shape[1],
        a.shape[1],
        a.shape[2],
        n_output_samples,
    )

    transmitter_ranges = (
        range(a.shape[3]) if normalize else (slice(None),)
    )
    for transmitter in transmitter_ranges:
        tx_slice = (
            slice(transmitter, transmitter + 1)
            if isinstance(transmitter, int)
            else transmitter
        )
        tx_a = a[:, :, :, tx_slice]
        tx_pulse = pulse[:, :, :, tx_slice]
        cached_taps = []
        energy = None
        for start in range(0, n_output_samples, time_chunk_size):
            taps = torch.einsum(
                "...pc,...pl->...cl",
                tx_a[..., start : start + time_chunk_size],
                tx_pulse,
            )
            cached_taps.append((start, taps))
            if normalize:
                partial = taps.abs().square().sum(dim=(-1, -2))
                energy = partial if energy is None else energy + partial

        scale = None
        if normalize:
            energy = energy.mean(dim=(2, 4), keepdim=True) / float(
                n_output_samples
            )
            scale = torch.complex(energy.sqrt(), torch.zeros_like(energy))
            scale = scale.unsqueeze(-1).unsqueeze(-1)

        for start, taps in cached_taps:
            stop = start + taps.shape[-2]
            if scale is not None:
                taps = torch.where(scale.abs() > 0, taps / scale, taps)
            output[..., start:stop] += (
                taps.unsqueeze(1)
                * gathered[:, :, :, :, tx_slice, :, start:stop, :]
            ).sum(dim=-1).sum(dim=5).sum(dim=4)
    return output


class SionnaTR38901System:
    """Sionna 2.1 time-domain uplink for paper-scale training/evaluation."""

    def __init__(self, config: EqDeepRxConfig, device: torch.device | str = "cpu"):
        try:
            from sionna.phy.channel import (
                ApplyTimeChannel,
                cir_to_ofdm_channel,
                cir_to_time_channel,
                gen_single_sector_topology_interferers,
                subcarrier_frequencies,
                time_lag_discrete_time_channel,
            )
            from sionna.phy.channel.tr38901 import CDL, PanelArray, UMa, UMi
            from sionna.phy.ofdm import OFDMDemodulator, OFDMModulator
        except ImportError as exc:
            raise RuntimeError(
                "Sionna 2.1 is required for the sionna_tr38901 backend"
            ) from exc

        self.config = config
        self.device = torch.device(device)
        self.sionna_device = (
            f"cuda:{torch.cuda.current_device()}"
            if self.device.type == "cuda" and self.device.index is None
            else str(self.device)
        )
        if config.n_fft < config.n_subcarriers:
            raise ValueError("n_fft must be at least n_subcarriers")

        self._ApplyTimeChannel = ApplyTimeChannel
        self._CDL = CDL
        self._PanelArray = PanelArray
        self._UMa = UMa
        self._UMi = UMi
        self._cir_to_ofdm_channel = cir_to_ofdm_channel
        self._cir_to_time_channel = cir_to_time_channel
        self._gen_topology = gen_single_sector_topology_interferers
        self._subcarrier_frequencies = subcarrier_frequencies
        self._system_level_channels = {}

        self.sample_rate_hz = config.sample_rate_hz
        self.l_min, self.l_max = time_lag_discrete_time_channel(
            self.sample_rate_hz,
            maximum_delay_spread=config.maximum_channel_delay_s,
        )
        self.l_tot = self.l_max - self.l_min + 1
        self._modulator = OFDMModulator(
            cyclic_prefix_length=config.cyclic_prefix, device=self.sionna_device
        )
        self._demodulator = OFDMDemodulator(
            fft_size=config.n_fft,
            l_min=self.l_min,
            cyclic_prefix_length=config.cyclic_prefix,
            device=self.sionna_device,
        )

        if config.n_rx_antennas % 2:
            polarization = "single"
            elements = config.n_rx_antennas
        else:
            polarization = "dual"
            elements = config.n_rx_antennas // 2
        rows = 2 if elements >= 4 and elements % 2 == 0 else 1
        columns = elements // rows
        self._bs_array = PanelArray(
            num_rows_per_panel=rows,
            num_cols_per_panel=columns,
            polarization=polarization,
            polarization_type="cross",
            antenna_pattern="38.901",
            carrier_frequency=config.carrier_frequency_hz,
            device=self.sionna_device,
        )
        self._ut_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=config.carrier_frequency_hz,
            device=self.sionna_device,
        )
        if self._bs_array.num_ant != config.n_rx_antennas:
            raise ValueError("unable to construct the configured BS antenna count")

    def _pilot_layout(self, n_layers: int, pilot_count: int) -> torch.Tensor:
        self.config.validate_layer_count(n_layers)
        if pilot_count not in (1, 2):
            raise ValueError("pilot_count must be 1 or 2")
        mask = torch.zeros(
            n_layers,
            self.config.n_subcarriers,
            self.config.n_ofdm_symbols,
            device=self.device,
        )
        symbols = self.config.receiver.dmrs_symbols[:pilot_count]
        spacing = self.config.receiver.pilot_spacing
        for layer in range(n_layers):
            carriers = torch.arange(
                layer % spacing,
                self.config.n_subcarriers,
                spacing,
                device=self.device,
            )
            mask[
                layer,
                carriers[:, None],
                torch.as_tensor(symbols, device=self.device)[None, :],
            ] = 1.0
        return mask

    def _resource_grid(
        self,
        *,
        batch_size: int,
        n_layers: int,
        pilot_count: int,
        generator: torch.Generator,
    ) -> Tuple[torch.Tensor, ...]:
        bps = bits_per_symbol(self.config.modulation)
        pilot_mask = self._pilot_layout(n_layers, pilot_count).unsqueeze(0).expand(
            batch_size, -1, -1, -1
        ).clone()
        pilot_ofdm_symbols = pilot_mask.any(dim=(1, 2))
        data_mask = (~pilot_ofdm_symbols[:, None, None, :]).expand(
            -1, 1, self.config.n_subcarriers, -1
        ).to(torch.float32).clone()
        bits = torch.randint(
            0,
            2,
            (
                batch_size,
                n_layers,
                self.config.n_subcarriers,
                self.config.n_ofdm_symbols,
                bps,
            ),
            generator=generator,
            device=self.device,
        ).to(torch.float32)
        data_symbols = qam_modulate(bits, self.config.modulation)
        pilot_bits = torch.randint(
            0,
            2,
            pilot_mask.shape + (2,),
            generator=generator,
            device=self.device,
        ).to(torch.float32)
        pilot_symbols = qam_modulate(pilot_bits, "QPSK") * pilot_mask
        transmitted = data_symbols * data_mask + pilot_symbols
        target_bits = (
            bits.permute(0, 1, 4, 2, 3).contiguous()
            * data_mask.unsqueeze(1)
        )
        return transmitted, pilot_symbols, pilot_mask, data_mask, target_bits

    def _serialize(self, grid: torch.Tensor) -> torch.Tensor:
        full = torch.zeros(
            *grid.shape[:2],
            self.config.n_ofdm_symbols,
            self.config.n_fft,
            dtype=grid.dtype,
            device=self.device,
        )
        start = (self.config.n_fft - self.config.n_subcarriers) // 2
        full[..., start : start + self.config.n_subcarriers] = grid.permute(
            0, 1, 3, 2
        )
        return self._modulator(full)

    def _system_level_cir(
        self,
        *,
        batch_size: int,
        n_layers: int,
        n_interferers: int,
        channel_model: str,
        speed_mps_range: Tuple[float, float],
        num_time_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scenario = channel_model.lower()
        model_type = self._UMa if scenario == "uma" else self._UMi
        cache_key = (scenario, batch_size, n_layers, n_interferers)
        channel = self._system_level_channels.get(cache_key)
        if channel is None:
            channel = model_type(
                carrier_frequency=self.config.carrier_frequency_hz,
                o2i_model="low",
                ut_array=self._ut_array,
                bs_array=self._bs_array,
                direction="uplink",
                enable_pathloss=False,
                enable_shadow_fading=False,
                device=self.sionna_device,
            )
            self._system_level_channels[cache_key] = channel
        topology = self._gen_topology(
            batch_size=batch_size,
            num_ut=n_layers,
            num_interferer=n_interferers,
            scenario=scenario,
            min_ut_velocity=float(speed_mps_range[0]),
            max_ut_velocity=float(speed_mps_range[1]),
            device=self.sionna_device,
        )
        channel.set_topology(*topology)
        return channel(
            num_time_samples=num_time_steps,
            sampling_frequency=self.sample_rate_hz,
        )

    def _cdl_cir(
        self,
        *,
        batch_size: int,
        n_transmitters: int,
        channel_model: str,
        speed_mps_range: Tuple[float, float],
        num_time_steps: int,
        generator: torch.Generator,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        model_name = channel_model.split("-", 1)[1].upper()
        batches_a = []
        batches_tau = []
        for _ in range(batch_size):
            links_a = []
            links_tau = []
            for _ in range(n_transmitters):
                delay_ns = torch.empty((), device=self.device).uniform_(
                    float(self.config.delay_spread_ns_range[0]),
                    float(self.config.delay_spread_ns_range[1]),
                    generator=generator,
                )
                channel = self._CDL(
                    model=model_name,
                    delay_spread=float(delay_ns.item()) * 1e-9,
                    carrier_frequency=self.config.carrier_frequency_hz,
                    ut_array=self._ut_array,
                    bs_array=self._bs_array,
                    direction="uplink",
                    min_speed=float(speed_mps_range[0]),
                    max_speed=float(speed_mps_range[1]),
                    device=self.sionna_device,
                )
                a, tau = channel(
                    batch_size=1,
                    num_time_steps=num_time_steps,
                    sampling_frequency=self.sample_rate_hz,
                )
                links_a.append(a)
                links_tau.append(tau)
            batches_a.append(torch.cat(links_a, dim=3))
            batches_tau.append(torch.cat(links_tau, dim=2))
        return torch.cat(batches_a, dim=0), torch.cat(batches_tau, dim=0)

    def _channel_truth(
        self, a: torch.Tensor, tau: torch.Tensor, n_layers: int
    ) -> torch.Tensor:
        symbol_length = self.config.n_fft + self.config.cyclic_prefix
        time_indices = (
            torch.arange(self.config.n_ofdm_symbols, device=self.device)
            * symbol_length
            + self.config.cyclic_prefix
        ).clamp_max(a.shape[-1] - 1)
        a_symbols = a[..., :n_layers, :, :, :].index_select(-1, time_indices)
        tau_symbols = tau[..., :n_layers, :]
        frequencies = self._subcarrier_frequencies(
            self.config.n_fft,
            self.config.subcarrier_spacing_khz * 1000.0,
            device=self.sionna_device,
        )
        response = self._cir_to_ofdm_channel(
            frequencies, a_symbols, tau_symbols, normalize=True
        )
        start = (self.config.n_fft - self.config.n_subcarriers) // 2
        response = response[
            :, 0, :, :n_layers, 0, :, start : start + self.config.n_subcarriers
        ]
        return response.permute(0, 1, 2, 4, 3).contiguous()

    @staticmethod
    def _shift_circular(waveform: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """Apply a per-slot timing offset to a continuously active interferer."""

        if offsets.shape != (waveform.shape[0],):
            raise ValueError("offsets must contain one value per batch sample")
        length = waveform.shape[-1]
        index_shape = (1,) * (waveform.ndim - 2) + (length,)
        offset_shape = (waveform.shape[0],) + (1,) * (waveform.ndim - 1)
        indices = (
            torch.arange(length, device=waveform.device).reshape(index_shape)
            - offsets.to(waveform.device).reshape(offset_shape)
        ) % length
        return torch.gather(waveform, -1, indices.expand_as(waveform))

    def generate_batch(
        self,
        *,
        batch_size: int,
        n_layers: int,
        pilot_count: int,
        snr_db: float | torch.Tensor,
        seed: int,
        add_interference: bool = False,
        sinr_db: float | None = None,
        channel_model: str = "UMa",
        speed_mps_range: Tuple[float, float] | None = None,
        return_true_channel: bool = True,
    ) -> SignalBatch:
        """Generate a complete time-domain TR 38.901 uncoded uplink batch."""

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.config.validate_layer_count(n_layers)
        normalized_model = channel_model.upper()
        if normalized_model not in {"UMA", "UMI", "CDL-C", "CDL-D"}:
            raise ValueError("channel_model must be UMa, UMi, CDL-C, or CDL-D")
        display_model = {
            "UMA": "UMa",
            "UMI": "UMi",
            "CDL-C": "CDL-C",
            "CDL-D": "CDL-D",
        }[normalized_model]
        speed_range = speed_mps_range or self.config.speed_mps_range
        if speed_range[0] < 0 or speed_range[1] < speed_range[0]:
            raise ValueError("speed_mps_range must be nonnegative and ordered")

        from sionna.phy import config as sionna_config

        torch.manual_seed(int(seed))
        sionna_config.seed = int(seed)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        (
            transmitted,
            pilot_symbols,
            pilot_mask,
            data_mask,
            target_bits,
        ) = self._resource_grid(
            batch_size=batch_size,
            n_layers=n_layers,
            pilot_count=pilot_count,
            generator=generator,
        )

        n_interferers = 1 if add_interference else 0
        if add_interference:
            interference_bits = torch.randint(
                0,
                2,
                (
                    batch_size,
                    1,
                    self.config.n_subcarriers,
                    self.config.n_ofdm_symbols,
                    bits_per_symbol(self.config.modulation),
                ),
                generator=generator,
                device=self.device,
            ).to(torch.float32)
            interference_grid = qam_modulate(
                interference_bits, self.config.modulation
            )
            all_grids = torch.cat((transmitted, interference_grid), dim=1)
        else:
            all_grids = transmitted
        waveform = self._serialize(all_grids)
        num_time_samples = waveform.shape[-1]
        num_channel_steps = num_time_samples + self.l_tot - 1

        if normalized_model in {"UMA", "UMI"}:
            a, tau = self._system_level_cir(
                batch_size=batch_size,
                n_layers=n_layers,
                n_interferers=n_interferers,
                channel_model=display_model,
                speed_mps_range=speed_range,
                num_time_steps=num_channel_steps,
            )
        else:
            a, tau = self._cdl_cir(
                batch_size=batch_size,
                n_transmitters=n_layers + n_interferers,
                channel_model=display_model,
                speed_mps_range=speed_range,
                num_time_steps=num_channel_steps,
                generator=generator,
            )

        x = waveform.unsqueeze(2)
        desired_x = x.clone()
        desired_x[:, n_layers:] = 0
        if add_interference:
            interference_x = x.clone()
            interference_x[:, :n_layers] = 0
            offsets = torch.randint(
                0,
                self.config.n_fft + self.config.cyclic_prefix,
                (batch_size,),
                generator=generator,
                device=self.device,
            )
            interference_x = self._shift_circular(interference_x, offsets)
            components = torch.stack((desired_x, interference_x), dim=1)
        else:
            components = desired_x.unsqueeze(1)
        component_waveforms = _apply_cir_time_channel_efficient(
            self.sample_rate_hz,
            components,
            a,
            tau,
            self.l_min,
            self.l_max,
            normalize=True,
        )
        desired_waveform = component_waveforms[:, 0, 0]
        if add_interference:
            interference_waveform = component_waveforms[:, 1, 0]
        else:
            interference_waveform = torch.zeros_like(desired_waveform)

        signal_power = desired_waveform.abs().square().mean(
            dim=(1, 2), keepdim=True
        ).clamp_min(1e-30)
        inr_db = 10.0 + 5.0 * torch.randn(
            batch_size, generator=generator, device=self.device
        )
        inr_linear = torch.pow(10.0, inr_db.view(-1, 1, 1) / 10.0)
        requested_snr = torch.as_tensor(
            snr_db, dtype=signal_power.dtype, device=self.device
        ).flatten()
        if requested_snr.numel() == 1:
            requested_snr = requested_snr.expand(batch_size)
        if requested_snr.numel() != batch_size:
            raise ValueError("snr_db must be scalar or have one value per sample")

        if sinr_db is None:
            noise_power = signal_power / torch.pow(
                10.0, requested_snr.view(-1, 1, 1) / 10.0
            )
        else:
            total_disturbance = signal_power / (10.0 ** (float(sinr_db) / 10.0))
            noise_power = total_disturbance / (
                1.0 + (inr_linear if add_interference else 0.0)
            )

        if add_interference:
            target_interference_power = noise_power * inr_linear
            measured_interference_power = interference_waveform.abs().square().mean(
                dim=(1, 2), keepdim=True
            ).clamp_min(1e-30)
            interference_waveform = interference_waveform * torch.sqrt(
                target_interference_power / measured_interference_power
            )
        else:
            target_interference_power = torch.zeros_like(noise_power)

        noise = torch.complex(
            torch.randn(
                desired_waveform.shape,
                generator=generator,
                device=self.device,
            ),
            torch.randn(
                desired_waveform.shape,
                generator=generator,
                device=self.device,
            ),
        ) * torch.sqrt(noise_power / 2.0)
        received_waveform = desired_waveform + interference_waveform + noise
        received_full = self._demodulator(received_waveform)
        start = (self.config.n_fft - self.config.n_subcarriers) // 2
        received = received_full[
            ..., start : start + self.config.n_subcarriers
        ].permute(0, 1, 3, 2).contiguous()
        true_channel = (
            self._channel_truth(a, tau, n_layers)
            if return_true_channel
            else torch.empty(0, dtype=a.dtype, device=self.device)
        )

        actual_snr = 10.0 * torch.log10(
            signal_power.flatten() / noise_power.flatten().clamp_min(1e-30)
        )
        actual_sinr = 10.0 * torch.log10(
            signal_power.flatten()
            / (noise_power + target_interference_power).flatten().clamp_min(1e-30)
        )
        snr_value: float | torch.Tensor
        sinr_value: float | torch.Tensor
        if batch_size == 1:
            snr_value = float(actual_snr.item())
            sinr_value = float(actual_sinr.item())
        else:
            snr_value = actual_snr.detach()
            sinr_value = actual_sinr.detach()

        return SignalBatch(
            received=received,
            transmitted=transmitted,
            pilot_symbols=pilot_symbols,
            pilot_mask=pilot_mask,
            data_mask=data_mask,
            target_bits=target_bits,
            true_channel=true_channel,
            noise_variance=noise_power.flatten(),
            snr_db=snr_value,
            backend="sionna_tr38901_time_domain",
            interference_present=bool(add_interference),
            sinr_db=float(sinr_db) if sinr_db is not None else None,
            realized_sinr_db=sinr_value,
            channel_model=display_model,
            speed_mps_range=(float(speed_range[0]), float(speed_range[1])),
            sample_rate_hz=self.sample_rate_hz,
        )
