from dataclasses import replace

import torch

from eqdeeprx.config import paper_config
from eqdeeprx.signal import OFDMSystem, bits_per_symbol, qam_modulate


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
    assert torch.all(batch.data_mask + batch.pilot_mask <= 1.0)
    assert torch.isfinite(batch.received.real).all()


def test_ofdm_generation_is_deterministic_for_a_seed():
    config = _tiny_config()
    system = OFDMSystem(config)
    first = system.generate_batch(batch_size=1, n_layers=2, pilot_count=1, snr_db=12.0, seed=11)
    second = system.generate_batch(batch_size=1, n_layers=2, pilot_count=1, snr_db=12.0, seed=11)

    assert torch.equal(first.received, second.received)
    assert torch.equal(first.target_bits, second.target_bits)

