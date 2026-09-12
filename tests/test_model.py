from dataclasses import replace

import pytest
import torch

from eqdeeprx.config import paper_config
from eqdeeprx.layers import SubsampledResidualBlock, TimeMixer
from eqdeeprx.model import DenoiseNN, DetectorNN, DemapperNN, EqDeepRx
from eqdeeprx.receiver import estimate_raw_channel
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


def test_denoise_nn_accepts_compact_one_and_two_pilot_grids():
    denoise = DenoiseNN(widths=(8, 8, 8, 2), subsamples=(1, 2, 2, 1))
    for pilot_symbols in (1, 2):
        output = denoise(torch.randn(3, 2, 6, pilot_symbols))
        assert output.shape == (3, 2, 6, pilot_symbols)


def test_time_mixer_changes_only_the_selected_pilot_channels():
    mixer = TimeMixer(max_symbols=2, mix_channels=1)
    with torch.no_grad():
        mixer.conv.weight.zero_()
        mixer.conv.bias.zero_()
        mixer.conv.weight[0, 1, 0, 0] = 1.0
        mixer.conv.weight[1, 0, 0, 0] = 1.0
    x = torch.zeros(1, 2, 3, 2)
    x[:, 0, :, 0] = 1.0
    x[:, 0, :, 1] = 2.0
    x[:, 1] = 5.0
    y = mixer(x)
    assert torch.allclose(y[:, 0, :, 0], torch.full((1, 3), 3.0))
    assert torch.allclose(y[:, 0, :, 1], torch.full((1, 3), 3.0))
    assert torch.equal(y[:, 1], x[:, 1])


def test_time_mixer_preserves_channel_slots_between_one_and_two_dmrs():
    mixer = TimeMixer(max_symbols=2, mix_channels=2)
    one_dmrs = torch.randn(1, 2, 5, 1)
    two_dmrs_with_missing_second = torch.cat(
        (one_dmrs, torch.zeros_like(one_dmrs)), dim=-1
    )

    one_output = mixer(one_dmrs)
    two_output = mixer(two_dmrs_with_missing_second)

    assert torch.allclose(one_output, two_output[..., :1])


def test_subsampled_block_has_paper_kernel_order_and_no_batch_norm():
    block = SubsampledResidualBlock(4, 4, downsample=1, frequency_only=False)
    assert block.conv1.depthwise.kernel_size == (1, 13)
    assert block.conv2.depthwise.kernel_size == (13, 1)
    assert not any(isinstance(module, torch.nn.BatchNorm2d) for module in block.modules())


def test_one_model_instance_handles_all_paper_mimo_layer_counts():
    base = paper_config().with_modulation("16QAM")
    config = replace(
        base,
        n_subcarriers=16,
        n_fft=24,
        cyclic_prefix=4,
        n_rx_antennas=4,
        n_tx_antennas=4,
        model=replace(
            base.model,
            detector_channels=8,
            detector_sections=1,
            demapper_widths=(4, 4, 4, 8),
            denoise_widths=(8, 8, 8, 2),
            denoise_subsamples=(1, 2, 2, 1),
        ),
    )
    model = EqDeepRx(config)
    detector_parameter_ids = tuple(id(parameter) for parameter in model.detector.parameters())
    demapper_parameter_ids = tuple(id(parameter) for parameter in model.demapper.parameters())

    for layers in (2, 3, 4):
        model.zero_grad(set_to_none=True)
        batch = OFDMSystem(config).generate_batch(
            batch_size=1,
            n_layers=layers,
            pilot_count=1,
            snr_db=12.0,
            seed=100 + layers,
        )
        logits = model(batch.received, batch.pilot_symbols, batch.pilot_mask)
        logits.square().mean().backward()
        assert logits.shape == (1, layers, 8, 16, 14)
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        assert tuple(id(parameter) for parameter in model.detector.parameters()) == detector_parameter_ids
        assert tuple(id(parameter) for parameter in model.demapper.parameters()) == demapper_parameter_ids


def test_vectorized_denoiser_matches_independent_antenna_pair_calls():
    config = _tiny_config()
    batch = OFDMSystem(config).generate_batch(
        batch_size=2,
        n_layers=2,
        pilot_count=2,
        snr_db=15.0,
        seed=91,
    )
    model = EqDeepRx(config)
    raw = estimate_raw_channel(
        batch.received, batch.pilot_symbols, batch.pilot_mask
    )
    actual = model._denoise_channel(raw, batch.pilot_mask)
    expected = torch.zeros_like(raw)
    for sample in range(raw.shape[0]):
        for layer in range(raw.shape[2]):
            locations = torch.nonzero(
                batch.pilot_mask[sample, layer] > 0, as_tuple=False
            )
            frequencies = torch.unique(locations[:, 0], sorted=True)
            symbols = torch.unique(locations[:, 1], sorted=True)
            compact = raw[sample, :, layer][
                :, frequencies[:, None], symbols[None, :]
            ]
            denoised = model.denoise(
                torch.stack((compact.real, compact.imag), dim=1)
            )
            estimate = torch.complex(denoised[:, 0], denoised[:, 1])
            expected[sample, :, layer][
                :, frequencies[:, None], symbols[None, :]
            ] = estimate

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_vectorized_shared_detector_matches_independent_layer_calls():
    config = _tiny_config()
    batch = OFDMSystem(config).generate_batch(
        batch_size=2,
        n_layers=2,
        pilot_count=1,
        snr_db=15.0,
        seed=92,
    )
    model = EqDeepRx(config)
    actual, aux = model(
        batch.received, batch.pilot_symbols, batch.pilot_mask, return_aux=True
    )
    coordinates = model._coordinates(
        2,
        config.n_subcarriers,
        config.n_ofdm_symbols,
        batch.received.device,
        batch.received.real.dtype,
    )
    expected = []
    for layer in range(2):
        detector_input = torch.cat(
            (
                aux["lmmse"][:, layer].real.unsqueeze(1),
                aux["lmmse"][:, layer].imag.unsqueeze(1),
                aux["rzf"][:, layer].real.unsqueeze(1),
                aux["rzf"][:, layer].imag.unsqueeze(1),
                coordinates,
            ),
            dim=1,
        )
        features, _ = model.detector(detector_input)
        expected.append(model.demapper(features))

    assert torch.allclose(actual, torch.stack(expected, dim=1), atol=1e-6, rtol=1e-6)


def test_model_preserves_float32_boundaries_under_autocast():
    config = replace(_tiny_config(), n_subcarriers=16, n_fft=24)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    batch = OFDMSystem(config, device=device).generate_batch(
        batch_size=1,
        n_layers=2,
        pilot_count=1,
        snr_db=15.0,
        seed=93,
    )
    model = EqDeepRx(config).to(device)

    with torch.autocast(device_type=device.type, dtype=dtype):
        logits, aux = model(
            batch.received,
            batch.pilot_symbols,
            batch.pilot_mask,
            return_aux=True,
        )

    assert logits.dtype == torch.float32
    assert all(state.dtype == torch.float32 for state in aux["symbol_states"])
    assert all(state.dtype == torch.float32 for state in aux["detector_states"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP regression")
def test_demapper_is_finite_for_large_features_under_cuda_autocast():
    demapper = DemapperNN(in_channels=64, widths=(32, 32, 32, 8)).cuda()
    with torch.no_grad():
        for module in demapper.modules():
            if isinstance(module, torch.nn.Conv2d):
                module.weight.fill_(1.0)
                if module.bias is not None:
                    module.bias.zero_()
    features = torch.full((1, 64, 2, 2), 2_000.0, device="cuda")

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = demapper(features)

    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()
