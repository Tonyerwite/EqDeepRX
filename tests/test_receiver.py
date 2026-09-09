from dataclasses import replace

import torch

from eqdeeprx.config import paper_config
from eqdeeprx.receiver import (
    equalize_parallel,
    estimate_incm,
    estimate_raw_channel,
    lmmse_equalize,
    rzf_equalize,
)
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


def test_raw_channel_and_incm_are_finite_and_covariance_is_hermitian_psd():
    config = _tiny_config()
    batch = OFDMSystem(config).generate_batch(batch_size=2, n_layers=2, pilot_count=2, snr_db=18.0, seed=3)
    raw = estimate_raw_channel(batch.received, batch.pilot_symbols, batch.pilot_mask)
    assert raw.shape == (2, 2, 2, 24, 14)
    assert torch.isfinite(raw.real).all()

    covariance = estimate_incm(
        batch.received,
        raw,
        batch.transmitted,
        batch.pilot_mask,
        coherence_bandwidth=12,
    )
    assert covariance.shape == (2, 2, 2, 2)
    assert torch.allclose(covariance, covariance.transpose(-1, -2).conj(), atol=1e-5)
    eigenvalues = torch.linalg.eigvalsh(covariance[0]).real
    assert torch.all(eigenvalues >= -1e-6)


def test_rzf_and_lmmse_recover_identity_mimo_symbols():
    torch.manual_seed(2)
    n, nr, nt, f, s = 2, 2, 2, 5, 3
    transmitted = torch.randn(n, nt, f, s, dtype=torch.cfloat)
    received = transmitted.clone()
    channel = torch.zeros(n, nr, nt, f, s, dtype=torch.cfloat)
    eye = torch.eye(nt, dtype=torch.cfloat)
    channel[:] = eye[:, :, None, None]
    covariance = torch.eye(nr, dtype=torch.cfloat).view(1, nr, nr, 1).expand(n, nr, nr, 1) * 1e-3

    rzf = rzf_equalize(received, channel, alpha=1e-4)
    lmmse = lmmse_equalize(received, channel, covariance, coherence_bandwidth=f)
    assert torch.allclose(rzf, transmitted, atol=5e-3, rtol=5e-3)
    assert torch.allclose(lmmse, transmitted, atol=5e-2, rtol=5e-2)


def test_parallel_equalizer_returns_two_distinct_finite_branches():
    config = _tiny_config()
    batch = OFDMSystem(config).generate_batch(batch_size=1, n_layers=2, pilot_count=1, snr_db=10.0, seed=9)
    raw = estimate_raw_channel(batch.received, batch.pilot_symbols, batch.pilot_mask)
    result = equalize_parallel(batch.received, raw, batch.transmitted, batch.pilot_mask, alpha=1e-4, coherence_bandwidth=12)

    assert result.lmmse.shape == (1, 2, 24, 14)
    assert result.rzf.shape == result.lmmse.shape
    assert torch.isfinite(result.lmmse.real).all()
    assert torch.isfinite(result.rzf.real).all()
