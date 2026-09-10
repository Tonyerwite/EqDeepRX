from dataclasses import replace

import pytest
import torch

from eqdeeprx.config import paper_config
from eqdeeprx.receiver import (
    _interp_last_dimension,
    equalize_parallel,
    estimate_incm,
    estimate_raw_channel,
    interpolate_channel,
    lmmse_equalize,
    lmmse_equalize_with_variance,
    oas_complex_shrinkage,
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


def test_batched_linear_interpolation_uses_the_last_dimension():
    known = torch.tensor([1.0, 3.0, 5.0])
    query = torch.arange(7, dtype=torch.float32)
    offsets = torch.tensor([[0.0], [10.0]])
    values = offsets + 2.0 * known

    actual = _interp_last_dimension(values, known, query)

    assert actual.shape == (2, 7)
    assert torch.allclose(actual, offsets + 2.0 * query)


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


def test_lmmse_reports_post_equalizer_disturbance_variance():
    received = torch.zeros(1, 2, 1, 1, dtype=torch.complex64)
    channel = torch.eye(2, dtype=torch.complex64).view(1, 2, 2, 1, 1)
    covariance = torch.diag(torch.tensor([0.25, 1.0])).to(torch.complex64)
    covariance = covariance.view(1, 2, 2, 1)

    equalized, variance = lmmse_equalize_with_variance(
        received,
        channel,
        covariance,
        coherence_bandwidth=1,
    )

    assert equalized.shape == (1, 2, 1, 1)
    assert variance.shape == equalized.shape
    assert torch.allclose(variance.flatten(), torch.tensor([0.25, 1.0]))


def test_lmmse_reuses_the_incm_inverse_across_each_frequency_band(monkeypatch):
    torch.manual_seed(17)
    batch, nr, nt, carriers, symbols = 1, 4, 2, 8, 3
    channel = torch.randn(
        batch, nr, nt, carriers, symbols, dtype=torch.complex64
    )
    received = torch.randn(batch, nr, carriers, symbols, dtype=torch.complex64)
    matrix = torch.randn(batch, 2, nr, nr, dtype=torch.complex64)
    covariance = matrix @ matrix.conj().transpose(-1, -2)
    covariance = covariance + 0.1 * torch.eye(nr, dtype=torch.complex64)
    covariance = covariance.permute(0, 2, 3, 1).contiguous()
    solved_shapes = []
    original_solve = torch.linalg.solve

    def recording_solve(left, right, *args, **kwargs):
        solved_shapes.append(tuple(left.shape))
        return original_solve(left, right, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "solve", recording_solve)
    equalized = lmmse_equalize(
        received, channel, covariance, coherence_bandwidth=4
    )

    assert torch.isfinite(equalized.real).all()
    assert not any(
        shape[-1] == nr and shape[1:3] == (carriers, symbols)
        for shape in solved_shapes
    )
    assert any(shape[-1] == nt for shape in solved_shapes)


def test_low_dimension_lmmse_matches_the_paper_direct_form():
    torch.manual_seed(31)
    batch, nr, nt, carriers, symbols = 2, 5, 3, 7, 2
    channel = torch.randn(
        batch, nr, nt, carriers, symbols, dtype=torch.complex64
    )
    received = torch.randn(batch, nr, carriers, symbols, dtype=torch.complex64)
    matrix = torch.randn(batch, 2, nr, nr, dtype=torch.complex64)
    covariance = matrix @ matrix.conj().transpose(-1, -2)
    covariance = covariance + 0.25 * torch.eye(nr, dtype=torch.complex64)
    covariance = covariance.permute(0, 2, 3, 1).contiguous()

    actual = lmmse_equalize(
        received, channel, covariance, coherence_bandwidth=4
    )
    expected = torch.empty_like(actual)
    for sample in range(batch):
        for carrier in range(carriers):
            band = carrier // 4
            disturbance = covariance[sample, :, :, band]
            for symbol in range(symbols):
                h = channel[sample, :, :, carrier, symbol]
                direct = h.conj().transpose(-1, -2) @ torch.linalg.solve(
                    h @ h.conj().transpose(-1, -2) + disturbance,
                    torch.eye(nr, dtype=h.dtype),
                )
                gain = torch.diagonal(direct @ h)
                expected[sample, :, carrier, symbol] = (
                    direct @ received[sample, :, carrier, symbol]
                ) / gain

    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_parallel_equalizer_returns_two_distinct_finite_branches():
    config = _tiny_config()
    batch = OFDMSystem(config).generate_batch(batch_size=1, n_layers=2, pilot_count=1, snr_db=10.0, seed=9)
    raw = estimate_raw_channel(batch.received, batch.pilot_symbols, batch.pilot_mask)
    result = equalize_parallel(batch.received, raw, batch.transmitted, batch.pilot_mask, alpha=1e-4, coherence_bandwidth=12)

    assert result.lmmse.shape == (1, 2, 24, 14)
    assert result.rzf.shape == result.lmmse.shape
    assert torch.isfinite(result.lmmse.real).all()
    assert torch.isfinite(result.rzf.real).all()


def test_complex_oas_uses_the_complex_gaussian_closed_form():
    samples = torch.tensor(
        [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 2.0 + 0.0j], [1.0 + 1.0j, 0.0 + 0.0j]],
        dtype=torch.cfloat,
    )
    covariance = samples.conj().transpose(0, 1) @ samples / samples.shape[0]
    p = covariance.shape[-1]
    trace = covariance.diagonal().real.sum()
    gamma = p * torch.trace(covariance @ covariance).real / trace.square()
    expected = ((p - gamma / p) / ((samples.shape[0] - 1 / p) * (gamma - 1))).clamp(0.0, 1.0)
    assert oas_complex_shrinkage(covariance, samples.shape[0]).item() == pytest.approx(expected.item())


def test_incm_is_hermitian_psd_after_oas_shrinkage():
    config = _tiny_config()
    batch = OFDMSystem(config).generate_batch(batch_size=1, n_layers=2, pilot_count=2, snr_db=12.0, seed=8)
    raw = estimate_raw_channel(batch.received, batch.pilot_symbols, batch.pilot_mask)
    covariance = estimate_incm(batch.received, raw, batch.transmitted, batch.pilot_mask)
    assert torch.allclose(covariance, covariance.conj().transpose(1, 2), atol=1e-5)
    eigenvalues = torch.linalg.eigvalsh(covariance.permute(0, 3, 1, 2))
    assert torch.all(eigenvalues >= -1e-5)


def test_incm_preserves_complex_spatial_covariance_orientation():
    samples = torch.tensor(
        [[1.0 + 1.0j, 2.0 - 1.0j], [0.5 - 2.0j, -1.0 + 0.25j], [2.0 + 0.5j, 0.25 + 1.5j]],
        dtype=torch.complex64,
    )
    received = samples.transpose(0, 1).reshape(1, 2, 3, 1)
    channel = torch.zeros(1, 2, 1, 3, 1, dtype=torch.complex64)
    transmitted = torch.ones(1, 1, 3, 1, dtype=torch.complex64)
    pilot_mask = torch.ones(1, 1, 3, 1)

    actual = estimate_incm(
        received,
        channel,
        transmitted,
        pilot_mask,
        coherence_bandwidth=3,
    )[0, :, :, 0]
    scm = samples.transpose(0, 1) @ samples.conj() / samples.shape[0]
    rho = oas_complex_shrinkage(scm, samples.shape[0])
    target = torch.trace(scm).real / 2.0 * torch.eye(2, dtype=torch.complex64)
    expected = (1.0 - rho) * scm + rho * target

    assert torch.allclose(actual, expected, atol=1e-6)


def test_incm_does_not_copy_pilot_indices_to_the_host(monkeypatch):
    config = _tiny_config()
    batch = OFDMSystem(config).generate_batch(
        batch_size=2,
        n_layers=2,
        pilot_count=2,
        snr_db=12.0,
        seed=81,
    )
    raw = estimate_raw_channel(
        batch.received, batch.pilot_symbols, batch.pilot_mask
    )

    def reject_host_copy(_self):
        raise AssertionError("INCM estimation must stay tensorized")

    monkeypatch.setattr(torch.Tensor, "tolist", reject_host_copy)
    estimate_incm(
        batch.received,
        raw,
        batch.transmitted,
        batch.pilot_mask,
        coherence_bandwidth=12,
    )


def test_channel_interpolation_does_not_copy_pilot_indices_to_the_host(
    monkeypatch,
):
    config = _tiny_config()
    batch = OFDMSystem(config).generate_batch(
        batch_size=2,
        n_layers=2,
        pilot_count=2,
        snr_db=12.0,
        seed=82,
    )
    raw = estimate_raw_channel(
        batch.received, batch.pilot_symbols, batch.pilot_mask
    )

    def reject_host_copy(_self):
        raise AssertionError("channel interpolation must stay tensorized")

    monkeypatch.setattr(torch.Tensor, "tolist", reject_host_copy)
    interpolated = interpolate_channel(raw, batch.pilot_mask)

    assert interpolated.shape == raw.shape
