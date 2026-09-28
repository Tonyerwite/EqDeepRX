from dataclasses import replace

import pytest

from eqdeeprx.config import FIGURE6A_SINR_POINTS, paper_config


def test_paper_config_uses_eqdeeprx_not_deeprx_signal_defaults():
    config = paper_config()

    assert config.subcarrier_spacing_khz == 30
    assert config.n_subcarriers == 192
    assert config.n_ofdm_symbols == 14
    assert config.n_rx_antennas == 16
    assert config.layer_counts == (2, 3, 4)
    assert config.training_channel == "UMa"
    assert config.validation_channels == ("UMa", "CDL-C", "CDL-D", "UMi")
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
    assert config.training.inr_mean_db == pytest.approx(10.0)
    assert config.training.inr_std_db == pytest.approx(5.0)
    assert config.ue_tx_antennas == 1


def test_paper_config_rejects_unsupported_layer_count_and_modulation():
    config = paper_config()

    with pytest.raises(ValueError, match="layer"):
        config.validate_layer_count(5)
    with pytest.raises(ValueError, match="modulation"):
        config.with_modulation("QPSK")


def test_eight_bit_network_entry_accepts_256qam():
    config = paper_config().with_modulation("256QAM")

    assert config.modulation == "256QAM"
    assert config.model.max_bits == 8


def test_paper_figure6a_protocol_is_frozen():
    config = paper_config()

    assert config.evaluation.channel_model == "CDL-C"
    assert config.evaluation.speed_mps_range == (10.0, 15.0)
    assert config.evaluation.sinr_db_points == FIGURE6A_SINR_POINTS
    assert config.evaluation.interfering_ues == 1
    assert config.evaluation.pilot_counts == (1, 2)
    assert config.evaluation.validation_samples == 32_000
    assert config.evaluation.sinr_bin_width_db == 1.0
    assert config.receiver.baseline_smoothing_window == 5
    assert not hasattr(config.receiver, "sample_rate_hz")


def test_ofdm_sample_rate_is_consistent_with_fft_and_scs():
    config = paper_config()

    assert config.sample_rate_hz == config.n_fft * config.subcarrier_spacing_khz * 1000.0
    assert config.sample_rate_hz == 7_680_000.0
    assert config.maximum_channel_delay_s >= 13.78e-6


def test_cdl_delay_spread_default_uses_paper_validation_range():
    config = paper_config()

    assert config.channel.cdl_delay_spread_mode == "uniform_10_1100ns"
    assert config.cdl_delay_spread_ns == pytest.approx(300.0)
    assert config.cdl_delay_spread_ns != config.delay_spread_ns_range[1]


def test_long_training_requires_sionna_backend():
    config = paper_config()

    assert config.training.backend == "sionna_tr38901"
    assert config.training.layer_counts == (2, 3, 4)


def test_unpublished_implementation_choices_are_explicit_and_frozen_by_default():
    config = paper_config()

    assert config.channel.uma_delay_spread_mode == "native"
    assert config.channel.cdl_delay_spread_mode == "uniform_10_1100ns"
    assert config.channel.cir_discretization == "sinc"
    assert config.channel.cir_normalization is True
    assert config.channel.enable_pathloss is False
    assert config.channel.enable_shadow_fading is False
    assert config.channel.interferer_timing == "random_symbol"
    assert config.training.amp_dtype == "bfloat16"
    assert config.training.lamb_bias_correction is True
    assert config.training.vcl_attachment == "all"
    assert config.training.symbol_loss_reduction == "sum"


def test_time_mixer_channel_count_must_be_positive():
    with pytest.raises(ValueError, match="time_mixer_channels"):
        replace(
            paper_config(),
            model=replace(paper_config().model, time_mixer_channels=0),
        )
