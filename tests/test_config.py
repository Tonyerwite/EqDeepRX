import pytest

from eqdeeprx.config import paper_config


def test_paper_config_uses_eqdeeprx_not_deeprx_signal_defaults():
    config = paper_config()

    assert config.subcarrier_spacing_khz == 30
    assert config.n_subcarriers == 192
    assert config.n_ofdm_symbols == 14
    assert config.n_rx_antennas == 16
    assert config.layer_counts == (2, 3, 4)
    assert config.training_channel == "UMa"
    assert config.modulation == "64QAM"
    assert config.rzf_alpha == pytest.approx(1e-4)
    assert config.incm_coherence_bandwidth == 24


def test_paper_config_freezes_paper_network_and_training_defaults():
    config = paper_config()

    assert config.model.denoise_widths == (64, 64, 64, 2)
    assert config.model.denoise_subsamples == (1, 4, 2, 1)
    assert config.model.detector_channels == 64
    assert config.model.detector_sections == 4
    assert config.model.demapper_widths == (32, 32, 32, 8)
    assert config.training.batch_size == 112
    assert config.training.learning_rate == pytest.approx(4.4e-3)
    assert config.training.total_steps == 70_000
    assert config.training.symbol_loss_weight == pytest.approx(1e-5)
    assert config.training.layer_counts == (2, 3, 4)
    assert config.training.interference_probability == pytest.approx(0.5)


def test_paper_config_rejects_unsupported_layer_count_and_modulation():
    config = paper_config()

    with pytest.raises(ValueError, match="layer"):
        config.validate_layer_count(5)
    with pytest.raises(ValueError, match="modulation"):
        config.with_modulation("QPSK")
