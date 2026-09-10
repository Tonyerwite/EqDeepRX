from dataclasses import replace
import subprocess
import sys

import pytest
import torch

import eqdeeprx.training as training_module
from eqdeeprx.config import paper_config
from eqdeeprx.losses import (
    activation_statistics,
    eqdeeprx_loss,
    vcl_regularization,
    vcl_regularization_from_statistics,
)
from eqdeeprx.signal import OFDMSystem
from eqdeeprx.training import (
    Lamb,
    config_fingerprint,
    estimate_training_runtime,
    paper_learning_rate,
    train_steps,
    validate_full_training_request,
)
from eqdeeprx.model import EqDeepRx


def _tiny_config():
    return replace(
        paper_config().with_modulation("16QAM"),
        n_subcarriers=16,
        n_fft=24,
        cyclic_prefix=4,
        n_rx_antennas=2,
        n_tx_antennas=2,
        layer_counts=(2,),
        model=replace(paper_config().model, detector_channels=8, detector_sections=1, demapper_widths=(4, 4, 4, 4), denoise_widths=(8, 8, 8, 2), denoise_subsamples=(1, 2, 2, 1)),
    )


def test_loss_uses_positive_bit_zero_llrs_and_snr_weighted_symbol_term():
    target = torch.tensor([[[[[0.0, 1.0], [1.0, 0.0]]]]])
    logits = (1.0 - target * 2.0) * 8.0
    mask = torch.ones(1, 1, 2, 2)
    bits = torch.ones(1, 1, 1, 1)
    state = [torch.zeros(1, 1, 2, 2, 2)]
    symbols = torch.ones(1, 1, 2, 2, dtype=torch.cfloat)

    low = eqdeeprx_loss(logits, target, mask, bits, state, symbols, snr_linear=1.0, lambda_symbol=1e-2)
    high = eqdeeprx_loss(logits, target, mask, bits, state, symbols, snr_linear=15.0, lambda_symbol=1e-2)
    inverted = eqdeeprx_loss(-logits, target, mask, bits, [], None, snr_linear=1.0, lambda_symbol=0.0)

    assert low.item() < high.item()
    assert inverted.item() > low.item()


def test_vcl_regularization_penalizes_nonzero_mean_and_nonunit_variance():
    activations = torch.ones(2, 3, 4, 4)
    assert vcl_regularization(activations).item() > 0.0
    assert vcl_regularization(torch.randn(2, 3, 4, 4)).item() >= 0.0


def test_vcl_regularization_matches_paper_scaling():
    activations = torch.ones(2, 4, 3, 3)
    # Paper source: alpha * (||mean||^2 + ||variance - 1||^2) / C.
    expected = 1e-5 * (1.0 + 1.0)
    assert vcl_regularization(activations, alpha=1e-5).item() == pytest.approx(expected)


def test_symbol_loss_uses_squared_l2_norm_across_mimo_layers():
    logits = torch.zeros(1, 1, 1, 1, 1)
    target_bits = torch.zeros_like(logits)
    mask = torch.ones(1, 1, 1, 1)
    bit_mask = torch.ones(1)
    one_state = [torch.zeros(1, 1, 2, 1, 1)]
    one_target = torch.ones(1, 1, 1, 1, dtype=torch.complex64)
    two_state = [one_state[0].expand(-1, 2, -1, -1, -1).clone()]
    two_target = one_target.expand(-1, 2, -1, -1).clone()

    one = eqdeeprx_loss(
        logits,
        target_bits,
        mask,
        bit_mask,
        one_state,
        one_target,
        lambda_symbol=1.0,
    )
    two = eqdeeprx_loss(
        logits.expand(-1, 2, -1, -1, -1),
        target_bits.expand(-1, 2, -1, -1, -1),
        mask,
        bit_mask,
        two_state,
        two_target,
        lambda_symbol=1.0,
    )

    assert two.item() - one.item() == pytest.approx(1.0)


def test_chunked_vcl_has_same_value_and_gradient_as_full_batch():
    torch.manual_seed(8)
    full = torch.randn(6, 4, 3, 2, requires_grad=True)
    expected = vcl_regularization(full, alpha=1e-5)
    expected.backward()
    expected_gradient = full.grad.detach().clone()

    chunked = full.detach().clone().requires_grad_(True)
    statistics = activation_statistics(tuple(chunked.detach().chunk(3)))
    actual = sum(
        vcl_regularization_from_statistics(
            chunk,
            statistics,
            alpha=1e-5,
        )
        for chunk in chunked.chunk(3)
    )
    actual.backward()

    assert actual.item() == pytest.approx(expected.item(), rel=1e-6)
    assert torch.allclose(chunked.grad, expected_gradient, atol=1e-8, rtol=1e-5)


def test_paper_lamb_and_linear_schedule_contract():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = Lamb([parameter], lr=4.4e-3, betas=(0.9, 0.999), eps=1e-6)
    before = parameter.detach().clone()
    (parameter.square()).backward()
    optimizer.step()
    assert not torch.equal(before, parameter.detach())
    assert paper_learning_rate(0, total_steps=10, base_lr=1.0, warmup_steps=0) == 1.0
    assert paper_learning_rate(4, total_steps=10, base_lr=1.0, warmup_steps=0) == pytest.approx(5.0 / 9.0)
    assert paper_learning_rate(9, total_steps=10, base_lr=1.0, warmup_steps=0) == 0.0


def test_short_training_is_deterministic_and_writes_atomic_checkpoint(tmp_path):
    config = _tiny_config()
    torch.manual_seed(21)
    first_model = EqDeepRx(config)
    torch.manual_seed(21)
    second_model = EqDeepRx(config)
    first = train_steps(first_model, OFDMSystem(config), config, steps=1, batch_size=1, n_layers=2, pilot_count=1, snr_db=12.0, seed=12, output_path=tmp_path / "first.pt")
    second = train_steps(second_model, OFDMSystem(config), config, steps=1, batch_size=1, n_layers=2, pilot_count=1, snr_db=12.0, seed=12, output_path=tmp_path / "second.pt")

    assert first["steps"] == second["steps"] == 1
    assert first["losses"] == second["losses"]
    assert (tmp_path / "first.pt").exists()
    assert not (tmp_path / "first.pt.tmp").exists()


def test_microbatch_training_preserves_effective_batch_and_checkpoint_state(tmp_path):
    config = _tiny_config()
    output = tmp_path / "microbatch.pt"
    torch.manual_seed(31)
    model = EqDeepRx(config)

    history = train_steps(
        model,
        OFDMSystem(config),
        config,
        steps=1,
        batch_size=2,
        microbatch_size=1,
        n_layers=2,
        pilot_count=1,
        snr_db=12.0,
        seed=17,
        output_path=output,
    )
    checkpoint = torch.load(output, map_location="cpu", weights_only=False)

    assert history["steps"] == 1
    assert history["effective_batch_size"] == 2
    assert history["microbatch_size"] == 1
    assert checkpoint["next_step"] == 1
    assert "optimizer_state_dict" in checkpoint


def test_training_decouples_safe_channel_generation_from_model_microbatches():
    config = _tiny_config()

    class RecordingSystem:
        def __init__(self):
            self.system = OFDMSystem(config)
            self.batch_sizes = []

        def generate_batch(self, **kwargs):
            self.batch_sizes.append(kwargs["batch_size"])
            return self.system.generate_batch(**kwargs)

    class RecordingModel(EqDeepRx):
        def __init__(self):
            super().__init__(config)
            self.batch_sizes = []

        def forward(self, received, *args, **kwargs):
            self.batch_sizes.append(received.shape[0])
            return super().forward(received, *args, **kwargs)

    system = RecordingSystem()
    model = RecordingModel()
    history = train_steps(
        model,
        system,
        config,
        steps=1,
        batch_size=4,
        microbatch_size=4,
        generation_batch_size=2,
        n_layers=2,
        pilot_count=1,
        snr_db=12.0,
        seed=17,
    )

    assert system.batch_sizes == [2, 2]
    assert model.batch_sizes == [4, 4]
    assert history["effective_batch_size"] == 4
    assert history["microbatch_size"] == 4
    assert history["generation_batch_size"] == 2


def test_training_uses_cuda_amp_by_default_and_checkpoints_scaler(tmp_path):
    config = _tiny_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = tmp_path / "amp.pt"
    model = EqDeepRx(config)

    history = train_steps(
        model,
        OFDMSystem(config, device=device),
        config,
        steps=1,
        batch_size=1,
        microbatch_size=1,
        n_layers=2,
        pilot_count=1,
        snr_db=12.0,
        seed=18,
        output_path=output,
        device=device,
    )
    checkpoint = torch.load(output, map_location="cpu", weights_only=False)

    expected_amp = device.type == "cuda"
    assert history["amp_enabled"] is expected_amp
    assert checkpoint["amp_enabled"] is expected_amp
    assert "grad_scaler_state_dict" in checkpoint
    assert torch.isfinite(torch.tensor(history["losses"])).all()
    if expected_amp:
        assert checkpoint["grad_scaler_state_dict"]["scale"] == 1.0
        assert checkpoint["optimizer_state_dict"]["state"]
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )


def test_random_training_snr_is_sampled_per_example():
    config = _tiny_config()

    class RecordingSystem:
        def __init__(self):
            self.system = OFDMSystem(config)
            self.snr_arguments = []
            self.return_true_channel = []

        def generate_batch(self, **kwargs):
            self.snr_arguments.append(kwargs["snr_db"])
            self.return_true_channel.append(kwargs["return_true_channel"])
            return self.system.generate_batch(**kwargs)

    system = RecordingSystem()
    train_steps(
        EqDeepRx(config),
        system,
        config,
        steps=1,
        batch_size=4,
        microbatch_size=2,
        n_layers=2,
        pilot_count=1,
        seed=73,
    )

    assert len(system.snr_arguments) == 2
    assert all(isinstance(value, torch.Tensor) for value in system.snr_arguments)
    assert all(value.shape == (2,) for value in system.snr_arguments)
    assert torch.unique(torch.cat(system.snr_arguments)).numel() == 4
    assert system.return_true_channel == [False, False]


def test_runtime_estimate_scales_smoke_batch_to_effective_batch():
    estimate = estimate_training_runtime(
        smoke_step_seconds=10.0,
        smoke_batch_size=2,
        effective_batch_size=112,
        total_steps=70_000,
    )

    assert estimate["estimated_optimizer_step_seconds"] == pytest.approx(560.0)
    assert estimate["estimated_total_training_days"] == pytest.approx(
        560.0 * 70_000 / 86_400.0
    )


def test_paper_training_configuration_contract_covers_all_shared_model_cases():
    cases = training_module.paper_training_configurations(paper_config())

    assert len(cases) == 12
    assert set(cases) == {
        (layers, pilots, interference)
        for layers in (2, 3, 4)
        for pilots in (1, 2)
        for interference in (False, True)
    }


def test_tiny_training_cli_initializes_the_model_from_its_seed(tmp_path):
    checkpoints = [tmp_path / "seed-a.pt", tmp_path / "seed-b.pt"]
    for checkpoint in checkpoints:
        subprocess.run(
            [
                sys.executable,
                "scripts/train.py",
                "--tiny",
                "--steps",
                "1",
                "--batch-size",
                "1",
                "--microbatch-size",
                "1",
                "--seed",
                "94",
                "--device",
                "cpu",
                "--output",
                str(checkpoint),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    first = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    second = torch.load(checkpoints[1], map_location="cpu", weights_only=False)
    assert first["history"] == second["history"]
    assert all(
        torch.equal(first["model_state_dict"][name], second["model_state_dict"][name])
        for name in first["model_state_dict"]
    )


def test_training_resume_continues_at_the_saved_global_step(tmp_path):
    config = _tiny_config()
    output = tmp_path / "resume.pt"
    torch.manual_seed(37)
    train_steps(
        EqDeepRx(config),
        OFDMSystem(config),
        config,
        steps=1,
        batch_size=1,
        n_layers=2,
        pilot_count=1,
        snr_db=12.0,
        seed=19,
        output_path=output,
    )

    resumed_model = EqDeepRx(config)
    history = train_steps(
        resumed_model,
        OFDMSystem(config),
        config,
        steps=2,
        batch_size=1,
        n_layers=2,
        pilot_count=1,
        snr_db=12.0,
        seed=19,
        output_path=output,
        resume_path=output,
    )

    assert history["steps"] == 2
    assert len(history["losses"]) == 2


def test_training_resume_rejects_a_different_configuration(tmp_path):
    config = _tiny_config()
    output = tmp_path / "resume.pt"
    train_steps(
        EqDeepRx(config),
        OFDMSystem(config),
        config,
        steps=1,
        batch_size=1,
        n_layers=2,
        pilot_count=1,
        snr_db=12.0,
        seed=19,
        output_path=output,
    )
    changed = replace(config, training=replace(config.training, learning_rate=1e-3))

    with pytest.raises(ValueError, match="configuration does not match"):
        train_steps(
            EqDeepRx(changed),
            OFDMSystem(changed),
            changed,
            steps=2,
            batch_size=1,
            n_layers=2,
            pilot_count=1,
            snr_db=12.0,
            seed=19,
            resume_path=output,
        )


def test_full_training_gate_requires_standard_cuda_preflight(tmp_path):
    config = paper_config()
    report = tmp_path / "preflight.json"
    report.write_text(
        __import__("json").dumps(
            {
                "long_training_ready": True,
                "config_fingerprint": config_fingerprint(config),
                "approved_microbatch_size": 2,
                "approved_generation_batch_size": 2,
                "estimated_total_training_days": 2.5,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="confirmation"):
        validate_full_training_request(
            config,
            steps=70_000,
            batch_size=112,
            microbatch_size=2,
            device="cuda",
            confirm=False,
            cuda_available=True,
            preflight_report=report,
        )
    with pytest.raises(RuntimeError, match="CUDA"):
        validate_full_training_request(
            config,
            steps=70_000,
            batch_size=112,
            microbatch_size=2,
            device="cpu",
            confirm=True,
            cuda_available=True,
            preflight_report=report,
        )
    with pytest.raises(RuntimeError, match="effective batch"):
        validate_full_training_request(
            config,
            steps=70_000,
            batch_size=2,
            microbatch_size=2,
            device="cuda",
            confirm=True,
            cuda_available=True,
            preflight_report=report,
        )

    with pytest.raises(RuntimeError, match="microbatch"):
        validate_full_training_request(
            config,
            steps=70_000,
            batch_size=112,
            microbatch_size=3,
            device="cuda",
            confirm=True,
            cuda_available=True,
            preflight_report=report,
        )

    with pytest.raises(RuntimeError, match="generation batch"):
        validate_full_training_request(
            config,
            steps=70_000,
            batch_size=112,
            microbatch_size=2,
            generation_batch_size=1,
            device="cuda",
            confirm=True,
            cuda_available=True,
            preflight_report=report,
        )

    validate_full_training_request(
        config,
        steps=70_000,
        batch_size=112,
        microbatch_size=2,
        device="cuda",
        confirm=True,
        cuda_available=True,
        preflight_report=report,
    )


def test_full_training_gate_rejects_preflight_over_eight_day_budget(tmp_path):
    config = paper_config()
    report = tmp_path / "preflight.json"
    report.write_text(
        __import__("json").dumps(
            {
                "long_training_ready": True,
                "config_fingerprint": config_fingerprint(config),
                "approved_microbatch_size": 2,
                "approved_generation_batch_size": 2,
                "estimated_total_training_days": 8.01,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="eight-day runtime budget"):
        validate_full_training_request(
            config,
            steps=70_000,
            batch_size=112,
            microbatch_size=2,
            device="cuda",
            confirm=True,
            cuda_available=True,
            preflight_report=report,
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"n_layers": 2}, "layer"),
        ({"pilot_count": 1}, "DMRS"),
        ({"snr_db": 12.0}, "SNR"),
    ],
)
def test_full_training_gate_rejects_paper_distribution_overrides(
    tmp_path, override, message
):
    config = paper_config()
    report = tmp_path / "preflight.json"
    report.write_text(
        __import__("json").dumps(
            {
                "long_training_ready": True,
                "config_fingerprint": config_fingerprint(config),
                "approved_microbatch_size": 2,
                "approved_generation_batch_size": 2,
                "estimated_total_training_days": 2.5,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match=message):
        validate_full_training_request(
            config,
            steps=70_000,
            batch_size=112,
            microbatch_size=2,
            generation_batch_size=2,
            device="cuda",
            confirm=True,
            cuda_available=True,
            preflight_report=report,
            **override,
        )


def test_training_resume_rejects_a_different_run_signature(tmp_path):
    config = _tiny_config()
    output = tmp_path / "resume-signature.pt"
    train_steps(
        EqDeepRx(config),
        OFDMSystem(config),
        config,
        steps=1,
        batch_size=1,
        n_layers=2,
        pilot_count=1,
        snr_db=12.0,
        seed=19,
        output_path=output,
    )
    checkpoint = torch.load(output, map_location="cpu", weights_only=False)

    assert checkpoint["run_signature"] == {
        "batch_size": 1,
        "generation_batch_size": 1,
        "microbatch_size": 1,
        "n_layers": 2,
        "pilot_count": 1,
        "seed": 19,
        "snr_db": 12.0,
    }
    with pytest.raises(ValueError, match="run parameters"):
        train_steps(
            EqDeepRx(config),
            OFDMSystem(config),
            config,
            steps=2,
            batch_size=1,
            n_layers=2,
            pilot_count=1,
            snr_db=12.0,
            seed=20,
            resume_path=output,
        )


def test_training_resume_matches_an_uninterrupted_run(tmp_path):
    config = _tiny_config()
    output = tmp_path / "segmented.pt"

    torch.manual_seed(101)
    uninterrupted = EqDeepRx(config)
    uninterrupted_history = train_steps(
        uninterrupted,
        OFDMSystem(config),
        config,
        steps=2,
        batch_size=1,
        n_layers=2,
        pilot_count=1,
        snr_db=12.0,
        seed=23,
    )

    torch.manual_seed(101)
    segmented = EqDeepRx(config)
    train_steps(
        segmented,
        OFDMSystem(config),
        config,
        steps=1,
        batch_size=1,
        n_layers=2,
        pilot_count=1,
        snr_db=12.0,
        seed=23,
        output_path=output,
    )
    resumed = EqDeepRx(config)
    resumed_history = train_steps(
        resumed,
        OFDMSystem(config),
        config,
        steps=2,
        batch_size=1,
        n_layers=2,
        pilot_count=1,
        snr_db=12.0,
        seed=23,
        resume_path=output,
    )

    assert resumed_history == uninterrupted_history
    assert all(
        torch.equal(uninterrupted.state_dict()[name], resumed.state_dict()[name])
        for name in uninterrupted.state_dict()
    )
