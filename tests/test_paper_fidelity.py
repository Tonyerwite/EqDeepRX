from dataclasses import asdict

import pytest

from eqdeeprx.config import FIGURE6A_SINR_POINTS, paper_config
from eqdeeprx.evaluation import PAPER_FIGURE6A_SINR_POINTS, evaluate_paper_figure6a
from eqdeeprx.layers import TimeMixer
from eqdeeprx.model import EqDeepRx
from eqdeeprx.training import paper_learning_rate, paper_training_configurations


def test_table_dimensions_and_public_model_size_are_exact():
    config = paper_config()
    assert (
        config.n_subcarriers,
        config.n_ofdm_symbols,
        config.n_rx_antennas,
        config.n_tx_antennas,
        config.model.max_bits,
    ) == (192, 14, 16, 4, 8)
    assert EqDeepRx(config).count_parameters() == 115456


def test_table_training_and_receiver_parameters_are_exact():
    config = paper_config()
    assert config.layer_counts == (2, 3, 4)
    assert config.training.layer_counts == (2, 3, 4)
    assert config.training.batch_size == 112
    assert config.training.total_steps == 70_000
    assert config.training.learning_rate == pytest.approx(4.4e-3)
    assert config.training.symbol_loss_weight == pytest.approx(1e-5)
    assert config.training.vcl_alpha == pytest.approx(1e-5)
    assert config.receiver.rzf_alpha == pytest.approx(1e-4)
    assert config.receiver.incm_coherence_bandwidth == 24
    assert config.receiver.pilot_spacing == 4
    assert config.model.denoise_widths == (64, 64, 64, 2)
    assert config.model.denoise_subsamples == (1, 4, 2, 1)
    assert config.model.detector_channels == 64
    assert config.model.detector_sections == 4
    assert config.model.detector_subsample == 8
    assert config.model.demapper_widths == (32, 32, 32, 8)


def test_public_training_case_matrix_has_all_layer_pilot_interference_cases():
    cases = paper_training_configurations(paper_config())
    assert len(cases) == 12
    assert set(cases) == {
        (layers, pilots, interference)
        for layers in (2, 3, 4)
        for pilots in (1, 2)
        for interference in (False, True)
    }


def test_public_inr_aggregate_row_records_both_runtime_parameters():
    from scripts.paper_fidelity_audit import build_manifest

    source = "Lognormal INR with mean 10 dB and standard deviation 5 dB"
    manifest = build_manifest(source, paper_source="fixture.tex", config=paper_config())
    entry = {
        item["id"]: item for item in manifest["entries"]
    }["training.inr_lognormal_db"]

    assert entry["observed"] == (10.0, 5.0)
    assert "training.inr_mean_db" in entry["evidence"]
    assert "training.inr_std_db" in entry["evidence"]


def test_figure6a_uses_four_layers_and_one_sinr_source():
    assert evaluate_paper_figure6a.__kwdefaults__["n_layers"] == 4
    assert FIGURE6A_SINR_POINTS == PAPER_FIGURE6A_SINR_POINTS
    assert FIGURE6A_SINR_POINTS == (-5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0)


def test_time_mixer_baseline_is_two_channels_and_shape_safe():
    config = paper_config()
    assert config.model.time_mixer_channels == 2
    model = EqDeepRx(config)
    assert all(
        mixer.mix_channels == config.model.time_mixer_channels
        for mixer in model.denoise.mixers
    )
    mixer = TimeMixer(max_symbols=2, mix_channels=config.model.time_mixer_channels)
    assert mixer.mix_channels == 2
    assert mixer.max_symbols == 2


def test_unpublished_residual_projection_bias_matches_runtime_baseline():
    config = paper_config()
    assert config.model.residual_projection_bias is False
    from scripts.paper_fidelity_audit import build_manifest

    manifest = build_manifest(
        "standard 1x1 projection is used for the residual branch",
        paper_source="fixture.tex",
        config=config,
    )
    entry = {
        item["id"]: item for item in manifest["entries"]
    }["architecture.residual_projection_bias"]
    assert entry["observed"] is False
    assert entry["status"] == "implementation_choice"


def test_linear_schedule_reaches_zero_at_the_final_step():
    assert paper_learning_rate(0, total_steps=70_000, base_lr=4.4e-3) == pytest.approx(4.4e-3)
    assert paper_learning_rate(69_999, total_steps=70_000, base_lr=4.4e-3) == 0.0
