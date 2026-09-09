from dataclasses import replace

import torch

from eqdeeprx.config import paper_config
from eqdeeprx.model import DenoiseNN, DetectorNN, DemapperNN, EqDeepRx
from eqdeeprx.signal import OFDMSystem


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


def test_submodules_follow_table_i_shapes_and_residual_state_contract():
    denoise = DenoiseNN(widths=(8, 8, 8, 2), subsamples=(1, 2, 2, 1))
    detector = DetectorNN(channels=8, sections=2, max_bits=4)
    demapper = DemapperNN(in_channels=8, widths=(4, 4, 4, 4))
    x = torch.randn(3, 2, 24, 14)

    assert denoise(x).shape == x.shape
    features, states = detector(torch.randn(3, 6, 24, 14))
    assert features.shape == (3, 8, 24, 14)
    assert len(states) == 2
    assert all(state.shape == (3, 2, 24, 14) for state in states)
    assert demapper(features).shape == (3, 4, 24, 14)


def test_eqdeeprx_runs_both_equalizers_and_shared_layer_network():
    config = _tiny_config()
    batch = OFDMSystem(config).generate_batch(batch_size=2, n_layers=2, pilot_count=2, snr_db=20.0, seed=4)
    model = EqDeepRx(config)
    logits, aux = model(batch.received, batch.pilot_symbols, batch.pilot_mask, return_aux=True)

    assert logits.shape == (2, 2, 8, 24, 14)
    assert aux["lmmse"].shape == (2, 2, 24, 14)
    assert aux["rzf"].shape == aux["lmmse"].shape
    assert len(aux["symbol_states"]) == config.model.detector_sections
    assert torch.isfinite(logits).all()


def test_model_supports_a_different_frequency_width_and_has_gradients():
    config = replace(_tiny_config(), n_subcarriers=16, n_fft=24)
    batch = OFDMSystem(config).generate_batch(batch_size=1, n_layers=2, pilot_count=1, snr_db=15.0, seed=5)
    model = EqDeepRx(config)
    logits = model(batch.received, batch.pilot_symbols, batch.pilot_mask)
    loss = logits.square().mean()
    loss.backward()

    assert logits.shape == (1, 2, 8, 16, 14)
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
    assert 50_000 <= model.count_parameters() <= 300_000
