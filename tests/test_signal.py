from dataclasses import replace
import math

import torch

from eqdeeprx.config import paper_config
from eqdeeprx.signal import OFDMSystem, bits_per_symbol, qam_demapper_llr, qam_modulate


def test_qam_uses_3gpp_interleaved_iq_gray_labels():
    bits = torch.tensor(
        [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]],
        dtype=torch.float32,
    )

    symbols = qam_modulate(bits, "16QAM")
    expected = torch.tensor(
        [1 + 1j, 1 + 3j, 3 + 1j, -1 + 1j], dtype=torch.complex64
    ) / math.sqrt(10.0)

    assert torch.allclose(symbols, expected)


def test_max_log_llr_uses_paper_positive_bit_zero_convention():
    bits = torch.tensor([[0, 1, 1, 0]], dtype=torch.float32)
    symbols = qam_modulate(bits, "16QAM")
    llrs = qam_demapper_llr(symbols, 1e-3, "16QAM", max_bits=4)

    assert torch.equal((llrs[:, 0] < 0).to(bits.dtype), bits[0])


def test_256qam_uses_all_eight_network_outputs_and_round_trips_bits():
    bits = torch.randint(
        0,
        2,
        (7, 8),
        generator=torch.Generator().manual_seed(17),
    ).float()

    symbols = qam_modulate(bits, "256QAM")
    llrs = qam_demapper_llr(symbols, 1e-4, "256QAM", max_bits=8)

    assert llrs.shape == (8, 7)
    assert torch.equal((llrs.transpose(0, 1) < 0).float(), bits)


def _tiny_config():
    return replace(
        paper_config().with_modulation("16QAM"),
        n_subcarriers=24,
        n_fft=32,
        cyclic_prefix=4,
        n_rx_antennas=2,
        n_tx_antennas=2,
        layer_counts=(2,),
    )


def test_qam_bits_round_trip_and_ofdm_batch_shapes():
    config = _tiny_config()
    assert bits_per_symbol(config.modulation) == 4
    bits = torch.randint(0, 2, (2, 2, 24, 14, 4), generator=torch.Generator().manual_seed(4)).float()
    symbols = qam_modulate(bits, config.modulation)
    assert symbols.shape == (2, 2, 24, 14)

    batch = OFDMSystem(config).generate_batch(batch_size=2, n_layers=2, pilot_count=2, snr_db=35.0, seed=7)
    assert batch.received.shape == (2, 2, 24, 14)
    assert batch.transmitted.shape == (2, 2, 24, 14)
    assert batch.target_bits.shape == (2, 2, 4, 24, 14)
    assert batch.pilot_mask.shape == (2, 2, 24, 14)
    assert batch.data_mask.shape == (2, 1, 24, 14)
    pilot_symbols = torch.nonzero(
        batch.pilot_mask.any(dim=(0, 1, 2)), as_tuple=False
    ).flatten()
    assert torch.count_nonzero(batch.data_mask[..., pilot_symbols]) == 0
    assert torch.count_nonzero(batch.data_mask[..., :]) == 2 * 24 * 12
    assert torch.isfinite(batch.received.real).all()


def test_ofdm_noise_power_broadcasts_over_batch_and_rx_dimensions():
    """A batch size different from the antenna count must remain valid."""

    config = _tiny_config()
    batch = OFDMSystem(config).generate_batch(
        batch_size=3,
        n_layers=2,
        pilot_count=1,
        snr_db=torch.tensor([8.0, 12.0, 16.0]),
        seed=13,
    )
    assert batch.received.shape == (3, 2, 24, 14)
    assert batch.noise_variance.shape == (3,)
    assert torch.isfinite(batch.received.real).all()
    interfered = OFDMSystem(config).generate_batch(
        batch_size=3,
        n_layers=2,
        pilot_count=1,
        snr_db=torch.tensor([8.0, 12.0, 16.0]),
        seed=13,
        add_interference=True,
    )
    assert interfered.received.shape == (3, 2, 24, 14)
    assert torch.isfinite(interfered.received.real).all()


def test_ofdm_generation_is_deterministic_for_a_seed():
    config = _tiny_config()
    system = OFDMSystem(config)
    first = system.generate_batch(batch_size=1, n_layers=2, pilot_count=1, snr_db=12.0, seed=11)
    second = system.generate_batch(batch_size=1, n_layers=2, pilot_count=1, snr_db=12.0, seed=11)

    assert torch.equal(first.received, second.received)
    assert torch.equal(first.target_bits, second.target_bits)


def test_two_dmrs_symbols_reuse_the_same_qpsk_pilot_sequence():
    config = _tiny_config()
    batch = OFDMSystem(config).generate_batch(
        batch_size=1,
        n_layers=2,
        pilot_count=2,
        snr_db=12.0,
        seed=12,
    )
    symbols = config.receiver.dmrs_symbols[:2]
    active = batch.pilot_mask[0, :, :, symbols[0]] > 0
    first = batch.pilot_symbols[0, :, :, symbols[0]]
    second = batch.pilot_symbols[0, :, :, symbols[1]]
    assert torch.equal(first[active], second[active])


def test_fast_interference_timing_rolls_the_time_axis_only():
    """Fast and Sionna backends must apply timing offsets along time."""

    waveform = torch.arange(1, 1 + 2 * 3 * 5 * 1, dtype=torch.float32).reshape(
        2, 3, 5, 1
    )
    shifted = OFDMSystem._shift_interference_waveform(
        waveform, torch.tensor([1, 2])
    )
    expected = torch.stack(
        (
            torch.roll(waveform[0], shifts=1, dims=1),
            torch.roll(waveform[1], shifts=2, dims=1),
        )
    )
    assert torch.equal(shifted, expected)


def test_fast_interference_timing_modes_have_distinct_offset_contracts():
    base = _tiny_config()
    generator = torch.Generator(device="cpu").manual_seed(91)
    symbol_offsets = OFDMSystem(base)._interference_timing_offsets(
        batch_size=64, waveform_length=100, generator=generator
    )
    assert int(symbol_offsets.min()) >= 0
    assert int(symbol_offsets.max()) < base.n_fft + base.cyclic_prefix

    zero_config = replace(
        base,
        channel=replace(base.channel, interferer_timing="zero"),
    )
    zero_offsets = OFDMSystem(zero_config)._interference_timing_offsets(
        batch_size=8,
        waveform_length=100,
        generator=torch.Generator(device="cpu").manual_seed(91),
    )
    assert torch.equal(zero_offsets, torch.zeros(8, dtype=torch.long))

    sample_config = replace(
        base,
        channel=replace(base.channel, interferer_timing="random_sample"),
    )
    sample_offsets = OFDMSystem(sample_config)._interference_timing_offsets(
        batch_size=64,
        waveform_length=100,
        generator=torch.Generator(device="cpu").manual_seed(91),
    )
    assert int(sample_offsets.min()) >= 0
    assert int(sample_offsets.max()) < 100
