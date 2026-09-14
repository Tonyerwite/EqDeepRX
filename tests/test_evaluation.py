from dataclasses import replace
import inspect
import subprocess
import sys

import pytest
import torch

from eqdeeprx.config import paper_config
from eqdeeprx.evaluation import (
    PAPER_FIGURE6A_SINR_POINTS,
    evaluate_paper_figure6a,
    evaluate_uncoded_ber,
)
from eqdeeprx.model import EqDeepRx
from eqdeeprx.signal import OFDMSystem


def test_figure6a_uses_the_measured_safe_evaluation_batch_size():
    parameter = inspect.signature(evaluate_paper_figure6a).parameters[
        "evaluation_batch_size"
    ]

    assert parameter.default == 2


def test_figure6a_uses_the_paper_three_layer_reference_configuration():
    parameter = inspect.signature(evaluate_paper_figure6a).parameters[
        "n_layers"
    ]

    assert parameter.default == 3


def test_figure6a_uses_the_paper_two_db_sinr_grid():
    assert PAPER_FIGURE6A_SINR_POINTS == (
        -5.0,
        -3.0,
        -1.0,
        1.0,
        3.0,
        5.0,
        7.0,
        9.0,
        11.0,
        13.0,
    )


def test_uncoded_ber_evaluation_returns_five_finite_curves(tmp_path):
    base = paper_config().with_modulation("16QAM")
    config = replace(
        base,
        n_subcarriers=16,
        n_fft=24,
        cyclic_prefix=4,
        n_rx_antennas=2,
        n_tx_antennas=2,
        layer_counts=(2,),
        model=replace(base.model, detector_channels=8, detector_sections=1, demapper_widths=(4, 4, 4, 4), denoise_widths=(8, 8, 8, 2), denoise_subsamples=(1, 2, 2, 1)),
    )
    metrics = evaluate_uncoded_ber(EqDeepRx(config), OFDMSystem(config), config, snr_points=(0.0, 6.0), samples_per_point=1, n_layers=2, seed=4, output_dir=tmp_path)

    assert set(metrics["curves"]) == {"eqdeeprx_1_pilot", "eqdeeprx_2_pilots", "lmmse_1_pilot", "lmmse_2_pilots", "lmmse_known_channel"}
    assert all(len(values) == 2 for values in metrics["curves"].values())
    assert all(value == value and value >= 0.0 for values in metrics["curves"].values() for value in values)
    assert (tmp_path / "uncoded_ber_metrics.json").exists()


def test_fast_evaluator_identifies_non_paper_backend(tmp_path):
    base = paper_config().with_modulation("16QAM")
    config = replace(
        base,
        n_subcarriers=16,
        n_fft=24,
        cyclic_prefix=4,
        n_rx_antennas=2,
        n_tx_antennas=2,
        layer_counts=(2,),
        model=replace(
            base.model,
            detector_channels=8,
            detector_sections=1,
            demapper_widths=(4, 4, 4, 4),
            denoise_widths=(8, 8, 8, 2),
            denoise_subsamples=(1, 2, 2, 1),
        ),
    )
    metrics = evaluate_uncoded_ber(
        EqDeepRx(config),
        OFDMSystem(config),
        config,
        snr_points=(0.0,),
        samples_per_point=1,
        n_layers=2,
        seed=9,
        output_dir=tmp_path,
    )

    assert metrics["backend"] == "fast_ofdm"
    assert metrics["paper_figure6a_protocol"] is False


def test_figure6a_uses_random_snr_and_realized_sinr_bins(tmp_path):
    base = paper_config().with_modulation("16QAM")
    config = replace(
        base,
        n_subcarriers=16,
        n_fft=24,
        cyclic_prefix=4,
        n_rx_antennas=2,
        n_tx_antennas=2,
        layer_counts=(2,),
        training=replace(base.training, layer_counts=(2,)),
        model=replace(
            base.model,
            detector_channels=8,
            detector_sections=1,
            demapper_widths=(4, 4, 4, 8),
            denoise_widths=(8, 8, 8, 2),
            denoise_subsamples=(1, 2, 2, 1),
        ),
    )

    class RecordingSystem:
        def __init__(self):
            self.fast = OFDMSystem(config)
            self.calls = []

        def generate_batch(self, **kwargs):
            self.calls.append(kwargs)
            batch = self.fast.generate_batch(
                batch_size=kwargs["batch_size"],
                n_layers=kwargs["n_layers"],
                pilot_count=kwargs["pilot_count"],
                snr_db=kwargs["snr_db"],
                seed=kwargs["seed"],
                add_interference=kwargs["add_interference"],
            )
            return replace(
                batch,
                backend="sionna_tr38901_time_domain",
                interference_present=True,
                sinr_db=None,
                realized_sinr_db=torch.zeros(kwargs["batch_size"]),
                channel_model="CDL-C",
                speed_mps_range=(10.0, 15.0),
            )

    system = RecordingSystem()
    metrics = evaluate_paper_figure6a(
        EqDeepRx(config),
        system,
        config,
        sinr_points=(0.0,),
        validation_samples=2,
        evaluation_batch_size=2,
        n_layers=2,
        seed=41,
        output_dir=tmp_path,
    )

    assert len(system.calls) == 2
    assert all(call.get("sinr_db") is None for call in system.calls)
    assert all(call["add_interference"] is True for call in system.calls)
    assert all(isinstance(call["snr_db"], torch.Tensor) for call in system.calls)
    assert all(call["snr_db"].shape == (1,) for call in system.calls)
    assert metrics["validation_samples_total"] == 2
    assert metrics["validation_samples_by_pilot"] == {"1": 1, "2": 1}
    assert metrics["sinr_bin_sample_counts"] == {
        "1_pilot": [1],
        "2_pilots": [1],
    }
    assert all(values[0] is not None for values in metrics["curves"].values())
    assert (tmp_path / "figure6a_progress.json").exists()
    assert not (tmp_path / "figure6a_progress.json.tmp").exists()
    progress = __import__("json").loads(
        (tmp_path / "figure6a_progress.json").read_text(encoding="utf-8")
    )
    one_pilot_bits_per_sample = 2 * 4 * 16 * 13
    two_pilot_bits_per_sample = 2 * 4 * 16 * 12
    assert progress["bits"]["baseline_known_channel"][0] == (
        one_pilot_bits_per_sample + two_pilot_bits_per_sample
    )

    with pytest.raises(ValueError, match="progress does not match"):
        evaluate_paper_figure6a(
            EqDeepRx(config),
            system,
            config,
            sinr_points=(0.0,),
            validation_samples=3,
            evaluation_batch_size=2,
            n_layers=2,
            seed=41,
            output_dir=tmp_path,
            resume=True,
        )


def test_standard_cli_requires_a_trained_checkpoint_before_backend_setup(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_uncoded_ber.py",
            "--device",
            "cpu",
            "--validation-samples",
            "1",
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert "--checkpoint is required" in result.stderr


def test_standard_cli_rejects_checkpoint_from_a_different_modulation(tmp_path):
    checkpoint_config = paper_config().with_modulation("16QAM")
    checkpoint = tmp_path / "16qam.pt"
    torch.save(
        {
            "model_state_dict": EqDeepRx(checkpoint_config).state_dict(),
            "config": checkpoint_config,
        },
        checkpoint,
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_uncoded_ber.py",
            "--checkpoint",
            str(checkpoint),
            "--validation-samples",
            "1",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path / "evaluation"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert "checkpoint configuration does not match" in result.stderr


def test_training_and_evaluation_clis_expose_paper_configuration_entries():
    train_help = subprocess.run(
        [sys.executable, "scripts/train.py", "--help"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    evaluation_help = subprocess.run(
        [sys.executable, "scripts/evaluate_uncoded_ber.py", "--help"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    assert "--modulation" in train_help
    assert "--n-layers" in train_help
    assert "--pilot-count" in train_help
    assert "--generation-batch-size" in train_help
    assert "--modulation" in evaluation_help
    assert "--sinr-points" in evaluation_help
    assert "--snr-points" in evaluation_help
