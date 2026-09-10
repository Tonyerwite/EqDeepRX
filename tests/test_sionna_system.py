from dataclasses import replace
import inspect

import pytest
import torch

pytest.importorskip("sionna")

from eqdeeprx.config import paper_config
from eqdeeprx.evaluation import evaluate_paper_figure6a
from eqdeeprx.model import EqDeepRx
from eqdeeprx.sionna_system import (
    SionnaTR38901System,
    _apply_cir_time_channel_efficient,
    _cir_to_time_channel_efficient,
)


def test_memory_efficient_time_channel_matches_sionna_reference():
    from sionna.phy.channel import cir_to_time_channel

    torch.manual_seed(7)
    a = torch.randn(2, 1, 3, 2, 1, 4, 9, dtype=torch.complex64)
    tau = torch.rand(2, 1, 2, 4) * 2e-6

    expected = cir_to_time_channel(7.68e6, a, tau, -3, 8, normalize=True)
    actual = _cir_to_time_channel_efficient(
        7.68e6, a, tau, -3, 8, normalize=True
    )

    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_fused_time_channel_matches_sionna_for_multiple_signal_components():
    from sionna.phy.channel import ApplyTimeChannel, cir_to_time_channel

    torch.manual_seed(8)
    n_time_samples = 11
    l_min, l_max = -3, 8
    l_tot = l_max - l_min + 1
    a = torch.randn(
        2,
        1,
        3,
        2,
        1,
        4,
        n_time_samples + l_tot - 1,
        dtype=torch.complex64,
    )
    tau = torch.rand(2, 1, 2, 4) * 2e-6
    signals = torch.randn(
        2, 2, 2, 1, n_time_samples, dtype=torch.complex64
    )
    taps = cir_to_time_channel(
        7.68e6, a, tau, l_min, l_max, normalize=True
    )
    apply_channel = ApplyTimeChannel(n_time_samples, l_tot, device="cpu")
    expected = torch.stack(
        [apply_channel(signals[:, index], taps) for index in range(2)],
        dim=1,
    )

    actual = _apply_cir_time_channel_efficient(
        7.68e6,
        signals,
        a,
        tau,
        l_min,
        l_max,
        normalize=True,
        time_chunk_size=4,
    )

    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_fused_time_channel_discretizes_each_cir_coefficient_once(monkeypatch):
    torch.manual_seed(9)
    n_time_samples = 11
    l_min, l_max = -3, 8
    l_tot = l_max - l_min + 1
    a = torch.randn(
        2,
        1,
        3,
        2,
        1,
        4,
        n_time_samples + l_tot - 1,
        dtype=torch.complex64,
    )
    tau = torch.rand(2, 1, 2, 4) * 2e-6
    signals = torch.randn(
        2, 2, 2, 1, n_time_samples, dtype=torch.complex64
    )
    original_einsum = torch.einsum
    processed_coefficients = 0

    def observed_einsum(equation, *operands):
        nonlocal processed_coefficients
        if equation == "...pc,...pl->...cl":
            processed_coefficients += operands[0].numel()
        return original_einsum(equation, *operands)

    monkeypatch.setattr(torch, "einsum", observed_einsum)
    _apply_cir_time_channel_efficient(
        7.68e6,
        signals,
        a,
        tau,
        l_min,
        l_max,
        normalize=True,
        time_chunk_size=4,
    )

    assert processed_coefficients == a.numel()


def test_fused_time_channel_uses_the_measured_safe_chunk_size():
    parameter = inspect.signature(_apply_cir_time_channel_efficient).parameters[
        "time_chunk_size"
    ]

    assert parameter.default == 256


def _tiny_standard_config():
    base = paper_config().with_modulation("16QAM")
    return replace(
        base,
        n_subcarriers=16,
        n_fft=24,
        cyclic_prefix=4,
        n_rx_antennas=4,
        n_tx_antennas=4,
        layer_counts=(2, 3, 4),
    )


def test_sionna_uma_time_domain_batch_has_receiver_contract():
    config = _tiny_standard_config()
    system = SionnaTR38901System(config, device="cpu")

    batch = system.generate_batch(
        batch_size=1,
        n_layers=2,
        pilot_count=1,
        snr_db=12.0,
        seed=19,
        add_interference=True,
        channel_model="UMa",
        speed_mps_range=(0.0, 1.0),
    )

    assert batch.backend == "sionna_tr38901_time_domain"
    assert batch.interference_present is True
    assert batch.received.shape == (1, 4, 16, 14)
    assert batch.true_channel.shape == (1, 4, 2, 16, 14)
    assert batch.target_bits.shape == (1, 2, 4, 16, 14)
    assert batch.noise_variance.shape == (1,)
    assert batch.sample_rate_hz == pytest.approx(24 * 30_000.0)
    assert torch.isfinite(batch.received.real).all()
    assert torch.isfinite(batch.true_channel.real).all()
    assert 1e-2 < batch.true_channel.abs().square().mean().item() < 1e2


def test_system_level_channel_model_is_reused_across_online_batches():
    config = _tiny_standard_config()
    system = SionnaTR38901System(config, device="cpu")
    kwargs = dict(
        batch_size=1,
        n_layers=2,
        n_interferers=0,
        channel_model="UMa",
        speed_mps_range=(0.0, 1.0),
        num_time_steps=16,
    )

    system._system_level_cir(**kwargs)
    first = system._system_level_channels[("uma", 1, 2, 0)]
    system._system_level_cir(**kwargs)
    system._system_level_cir(
        **{
            **kwargs,
            "batch_size": 2,
            "n_layers": 3,
        }
    )

    assert system._system_level_channels[("uma", 1, 2, 0)] is first
    assert len(system._system_level_channels) == 2


def test_known_channel_uses_the_official_post_cp_symbol_sampling_point():
    config = _tiny_standard_config()
    system = SionnaTR38901System(config, device="cpu")
    symbol_length = config.n_fft + config.cyclic_prefix
    time_steps = config.n_ofdm_symbols * symbol_length + system.l_tot - 1
    values = torch.arange(time_steps, dtype=torch.float32)
    a = torch.complex(values, torch.zeros_like(values)).view(
        1, 1, 1, 1, 1, 1, -1
    )
    tau = torch.zeros(1, 1, 1, 1)
    system._subcarrier_frequencies = lambda *args, **kwargs: torch.zeros(
        config.n_fft
    )

    def expose_symbol_samples(frequencies, a_symbols, tau_symbols, normalize):
        return a_symbols.squeeze(-2).unsqueeze(-1).expand(
            *a_symbols.shape[:-2], a_symbols.shape[-1], frequencies.numel()
        )

    system._cir_to_ofdm_channel = expose_symbol_samples
    channel = system._channel_truth(a, tau, n_layers=1)
    expected = (
        torch.arange(config.n_ofdm_symbols) * symbol_length
        + config.cyclic_prefix
    )

    assert torch.equal(channel[0, 0, 0, 0].real, expected.float())


def test_sionna_cdl_batch_supports_paper_figure6a_sinr_control():
    config = _tiny_standard_config()
    system = SionnaTR38901System(config, device="cpu")

    batch = system.generate_batch(
        batch_size=1,
        n_layers=2,
        pilot_count=2,
        snr_db=30.0,
        sinr_db=3.0,
        seed=23,
        add_interference=True,
        channel_model="CDL-C",
        speed_mps_range=(10.0, 15.0),
    )

    assert batch.backend == "sionna_tr38901_time_domain"
    assert batch.sinr_db == pytest.approx(3.0)
    assert batch.channel_model == "CDL-C"
    assert batch.speed_mps_range == (10.0, 15.0)
    assert batch.realized_sinr_db == pytest.approx(3.0, abs=0.05)
    pilot_symbols = torch.nonzero(
        batch.pilot_mask.any(dim=(0, 1, 2)), as_tuple=False
    ).flatten()
    assert torch.count_nonzero(batch.data_mask[..., pilot_symbols]) == 0


def test_time_channel_window_covers_the_longest_configured_cdl_d_path():
    config = _tiny_standard_config()
    system = SionnaTR38901System(config, device="cpu")

    represented_delay = (system.l_max - 6) / system.sample_rate_hz

    assert represented_delay >= config.maximum_channel_delay_s
    assert config.maximum_channel_delay_s >= 12.525 * 1.1e-6


def test_interferer_timing_offset_preserves_continuous_slot_occupancy():
    waveform = torch.arange(1, 9, dtype=torch.float32).reshape(1, 1, 1, 8)

    shifted = SionnaTR38901System._shift_circular(
        waveform, torch.tensor([3])
    )

    assert torch.equal(shifted, torch.roll(waveform, shifts=3, dims=-1))
    assert torch.count_nonzero(shifted) == torch.count_nonzero(waveform)


def test_paper_figure6a_evaluator_uses_standard_interference_link(tmp_path):
    config = _tiny_standard_config()
    config = replace(
        config,
        layer_counts=(2,),
        training=replace(config.training, layer_counts=(2,)),
        model=replace(
            config.model,
            detector_channels=8,
            detector_sections=1,
            demapper_widths=(4, 4, 4, 8),
            denoise_widths=(8, 8, 8, 2),
            denoise_subsamples=(1, 2, 2, 1),
        ),
    )
    metrics = evaluate_paper_figure6a(
        EqDeepRx(config),
        SionnaTR38901System(config, device="cpu"),
        config,
        sinr_points=(3.0,),
        validation_samples=2,
        evaluation_batch_size=1,
        n_layers=2,
        seed=29,
        output_dir=tmp_path,
    )

    assert metrics["backend"] == "sionna_tr38901_time_domain"
    assert metrics["paper_figure6a_protocol"] is True
    assert metrics["channel_model"] == "CDL-C"
    assert metrics["speed_mps_range"] == [10.0, 15.0]
    assert metrics["interfering_ues"] == 1
    assert metrics["sinr_db"] == [3.0]
    assert metrics["validation_samples_total"] == 2
    assert metrics["validation_samples_by_pilot"] == {"1": 1, "2": 1}
    assert set(metrics["curves"]) == {
        "eqdeeprx_1_pilot",
        "eqdeeprx_2_pilots",
        "baseline_1_pilot",
        "baseline_2_pilots",
        "baseline_known_channel",
    }
