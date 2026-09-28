import json
import inspect
from dataclasses import replace

import torch

from eqdeeprx.config import paper_config
from scripts.paper_fidelity_audit import build_manifest
from scripts.digitize_paper_figure6a import build_reference
from scripts.run_unpublished_sensitivity import (
    DEFAULT_CANDIDATES,
    SENSITIVITY_SCHEMA_VERSION,
    _apply_reference_state,
    _candidate_system_seed,
    _reference_model_state,
    assess_validation_coverage,
    apply_candidate,
    build_sensitivity_summary,
    formal_baseline_value,
    load_existing_record,
    parse_candidate_values,
    summarize_candidate_decisions,
    validation_plan_fingerprint,
)
import scripts.run_unpublished_sensitivity as sensitivity
from eqdeeprx.model import EqDeepRx


def test_paper_manifest_separates_public_facts_from_choices():
    manifest = build_manifest(
        "Table II: No. of subcarriers 192\nNo. of OFDM symbols per slot 14\n"
        "SNR U(0dB,45dB)\n",
        paper_source="fixture.tex",
        config=paper_config(),
    )
    entries = {entry["id"]: entry for entry in manifest["entries"]}
    assert entries["dimensions.n_subcarriers"]["status"] == "public"
    assert entries["dimensions.n_subcarriers"]["matched"] is True
    assert entries["model.time_mixer_channels"]["status"] == "implementation_choice"
    assert entries["validation.cdl_delay_spread_ns"]["status"] == "implementation_choice"
    assert entries["validation.cdl_delay_spread_ns"]["observed"] == 300.0
    # The fixture intentionally contains only a subset of the paper rows;
    # missing source evidence must keep the formal gate closed.
    assert manifest["all_public_matched"] is False


def test_delay_spread_range_is_public_for_cdl_validation_only():
    manifest = build_manifest(
        "Table II: Delay spread & UMa & 10--1100 ns\\n",
        paper_source="fixture.tex",
        config=paper_config(),
    )
    entries = {entry["id"]: entry for entry in manifest["entries"]}
    assert entries["validation.cdl_delay_spread_range_ns"]["status"] == "public"
    assert entries["validation.cdl_delay_spread_range_ns"]["matched"] is True
    assert "training.uma_delay_spread_range_ns" not in entries
    assert entries["validation.cdl_delay_spread_ns"]["status"] == "implementation_choice"


def test_paper_manifest_records_two_runtime_equalizer_branches():
    manifest = build_manifest(
        "Table I: E=2 equalizer branches\n",
        paper_source="fixture.tex",
        config=paper_config(),
    )
    entry = {
        item["id"]: item for item in manifest["entries"]
    }["dimensions.equalizer_branches"]
    assert entry["observed"] == 2
    assert entry["evidence"].endswith("runtime equalizer branch count")


def test_sensitivity_parser_and_candidate_application_are_single_variable():
    assert parse_candidate_values("1,2,4", "time_mixer_channels") == [1, 2, 4]
    config = paper_config()
    candidate = apply_candidate(config, "time_mixer_channels", 4)
    assert candidate.model.time_mixer_channels == 4
    assert candidate.training.learning_rate == config.training.learning_rate
    assert set(DEFAULT_CANDIDATES) >= {
        "time_mixer_channels",
        "cdl_delay_spread_mode",
        "uma_delay_spread_mode",
        "cir_discretization",
        "cir_normalization",
        "pathloss",
        "shadow_fading",
        "interferer_timing",
        "lamb_beta1",
        "lamb_beta2",
        "lamb_eps",
        "weight_decay",
        "lamb_bias_correction",
        "amp_dtype",
        "vcl_attachment",
        "symbol_loss_reduction",
        "residual_projection_bias",
    }


def test_residual_projection_bias_is_a_single_model_choice():
    config = paper_config()
    candidate = apply_candidate(config, "residual_projection_bias", True)
    assert candidate.model.residual_projection_bias is True
    assert candidate.model.time_mixer_channels == config.model.time_mixer_channels


def test_cdl_delay_spread_mode_is_a_single_explicit_sensitivity_choice():
    assert parse_candidate_values(
        "uniform_10_1100ns,fixed_300ns", "cdl_delay_spread_mode"
    ) == ["uniform_10_1100ns", "fixed_300ns"]
    config = paper_config()
    candidate = apply_candidate(config, "cdl_delay_spread_mode", "fixed_500ns")
    assert candidate.cdl_delay_spread_ns == 500.0
    assert candidate.channel.cdl_delay_spread_mode == "fixed"
    assert candidate.delay_spread_ns_range == config.delay_spread_ns_range


def test_sensitivity_baseline_is_the_formal_config_not_candidate_order():
    config = paper_config()
    assert formal_baseline_value(config, "time_mixer_channels", [1, 2, 4]) == 2
    assert formal_baseline_value(config, "lamb_bias_correction", [True, False]) is True


def test_resumable_sensitivity_loader_rejects_stale_parameter_records(tmp_path):
    path = tmp_path / "candidate.json"
    path.write_text(
        json.dumps(
            {
                "variable": "amp_dtype",
                "value": "float32",
                "seed": 2026,
                "screening_steps": 20,
                "validation_samples": 400,
                "status": "complete",
                "sensitivity_schema_version": SENSITIVITY_SCHEMA_VERSION,
                "system_seed": _candidate_system_seed(2026),
                "validation_plan_fingerprint": validation_plan_fingerprint(
                    seed=2026, validation_samples=400
                ),
            }
        ),
        encoding="utf-8",
    )
    loaded = load_existing_record(
        path,
        variable="amp_dtype",
        value="float32",
        seed=2026,
        screening_steps=20,
        validation_samples=400,
    )
    assert loaded["status"] == "complete"
    assert load_existing_record(
        path,
        variable="amp_dtype",
        value="float32",
        seed=2026,
        screening_steps=21,
        validation_samples=400,
    ) is None

    stale = json.loads(path.read_text(encoding="utf-8"))
    stale["sensitivity_schema_version"] = SENSITIVITY_SCHEMA_VERSION - 1
    path.write_text(json.dumps(stale), encoding="utf-8")
    assert load_existing_record(
        path,
        variable="amp_dtype",
        value="float32",
        seed=2026,
        screening_steps=20,
        validation_samples=400,
    ) is None


def test_resumable_sensitivity_loader_checks_recorded_training_steps(tmp_path):
    path = tmp_path / "candidate.json"
    path.write_text(
        json.dumps(
            {
                "variable": "amp_dtype",
                "value": "float32",
                "seed": 2026,
                "screening_steps": 20,
                "training_steps": 20,
                "validation_samples": 400,
                "status": "complete",
                "sensitivity_schema_version": SENSITIVITY_SCHEMA_VERSION,
                "system_seed": _candidate_system_seed(2026),
                "validation_plan_fingerprint": validation_plan_fingerprint(
                    seed=2026, validation_samples=400
                ),
            }
        ),
        encoding="utf-8",
    )
    assert load_existing_record(
        path,
        variable="amp_dtype",
        value="float32",
        seed=2026,
        screening_steps=20,
        validation_samples=400,
        expected_training_steps=20,
    ) is not None
    assert load_existing_record(
        path,
        variable="amp_dtype",
        value="float32",
        seed=2026,
        screening_steps=20,
        validation_samples=400,
        expected_training_steps=2000,
    ) is None


def test_resumable_sensitivity_loader_requires_rng_and_validation_contract(tmp_path):
    path = tmp_path / "candidate.json"
    path.write_text(
        json.dumps(
            {
                "variable": "amp_dtype",
                "value": "float32",
                "seed": 2026,
                "screening_steps": 20,
                "training_steps": 20,
                "validation_samples": 400,
                "status": "complete",
                "finite": True,
                "sensitivity_schema_version": SENSITIVITY_SCHEMA_VERSION,
            }
        ),
        encoding="utf-8",
    )
    assert load_existing_record(
        path,
        variable="amp_dtype",
        value="float32",
        seed=2026,
        screening_steps=20,
        validation_samples=400,
        expected_training_steps=20,
    ) is None


def test_sensitivity_summary_marks_missing_matrix_records_incomplete():
    summary = build_sensitivity_summary(
        variables=("amp_dtype",),
        matrix={"amp_dtype": ["bfloat16", "float32"]},
        screening_seeds=(2026, 2027),
        confirmation_seed=2028,
        coverage_policy={"minimum_in_range_samples": 100},
        records=[],
    )
    assert summary["status"] == "incomplete"
    assert summary["expected_record_count"] == 6
    assert len(summary["missing_records"]) == 6


def test_figure_reference_never_invents_coordinates(tmp_path):
    reference = build_reference(tmp_path)
    assert reference["status"] == "unavailable"
    assert reference["coordinates"] == []


def test_sensitivity_coverage_rejects_empty_displayed_sinr_evidence():
    metrics = {
        "in_range_sample_count": 0,
        "in_range_sample_count_by_pilot": {"1": 0, "2": 0},
    }
    accepted, reason = assess_validation_coverage(
        metrics,
        minimum_in_range_samples=10,
        minimum_in_range_per_pilot=2,
    )
    assert accepted is False
    assert reason == "in-range samples 0 < required 10"


def test_sensitivity_reference_initialization_preserves_unchanged_parameters():
    base = paper_config().with_modulation("16QAM")
    base = replace(
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
    reference_config = apply_candidate(base, "time_mixer_channels", 1)
    reference_state, _ = _reference_model_state(reference_config, 2026)
    candidate_config = apply_candidate(base, "time_mixer_channels", 4)
    candidate = EqDeepRx(candidate_config)
    copied = _apply_reference_state(candidate, reference_state)
    assert copied > 0
    assert torch.equal(
        candidate.state_dict()["detector.project.weight"],
        reference_state["detector.project.weight"],
    )


def test_sensitivity_system_seed_is_isolated_from_model_initialization():
    """Candidate model shapes must not choose a different channel RNG stream."""

    assert _candidate_system_seed(2026) != 2026
    assert _candidate_system_seed(2026) == _candidate_system_seed(2026)
    assert _candidate_system_seed(2026) != _candidate_system_seed(2027)


def test_validation_plan_fingerprint_is_independent_of_candidate_config():
    first = validation_plan_fingerprint(seed=2026, validation_samples=400)
    second = validation_plan_fingerprint(seed=2026, validation_samples=400)
    other_seed = validation_plan_fingerprint(seed=2027, validation_samples=400)

    assert first == second
    assert first != other_seed


def test_validation_plan_fingerprint_binds_evaluation_contract():
    baseline = validation_plan_fingerprint(
        seed=2026,
        validation_samples=400,
        n_layers=4,
        evaluation_batch_size=2,
        channel_model="CDL-C",
        speed_mps_range=(10.0, 15.0),
    )
    assert baseline != validation_plan_fingerprint(
        seed=2026,
        validation_samples=400,
        n_layers=3,
        evaluation_batch_size=2,
        channel_model="CDL-C",
        speed_mps_range=(10.0, 15.0),
    )
    assert baseline != validation_plan_fingerprint(
        seed=2026,
        validation_samples=400,
        n_layers=4,
        evaluation_batch_size=4,
        channel_model="CDL-C",
        speed_mps_range=(10.0, 15.0),
    )


def test_sensitivity_run_accepts_and_forwards_evaluation_batch_size(monkeypatch):
    """The validation batch is explicit so speed changes cannot mix records."""

    assert "evaluation_batch_size" in inspect.signature(sensitivity.run_one).parameters
    config = replace(
        paper_config().with_modulation("16QAM"),
        n_subcarriers=16,
        n_fft=24,
        cyclic_prefix=4,
        n_rx_antennas=2,
        n_tx_antennas=2,
        layer_counts=(2,),
        training=replace(paper_config().training, layer_counts=(2,)),
        model=replace(
            paper_config().model,
            detector_channels=8,
            detector_sections=1,
            demapper_widths=(4, 4, 4, 4),
            denoise_widths=(8, 8, 8, 2),
            denoise_subsamples=(1, 2, 2, 1),
        ),
    )
    captured = {}

    def fake_train(*args, **kwargs):
        return {"steps": 1, "losses": [1.0], "bers": [0.1]}

    def fake_evaluate(*args, **kwargs):
        captured["evaluation_batch_size"] = kwargs["evaluation_batch_size"]
        return {
            "curves": {"eqdeeprx_1_pilot": [0.2]},
            "curve_error_counts": {"eqdeeprx_1_pilot": [2.0]},
            "curve_bit_counts": {"eqdeeprx_1_pilot": [10.0]},
            "in_range_sample_count": 100,
            "in_range_sample_count_by_pilot": {"1": 50, "2": 50},
        }

    monkeypatch.setattr(sensitivity, "train_steps", fake_train)
    monkeypatch.setattr(sensitivity, "_evaluate", fake_evaluate)
    sensitivity.run_one(
        config,
        variable="time_mixer_channels",
        value=2,
        seed=2026,
        backend="fast",
        device="cpu",
        screening_steps=1,
        batch_size=2,
        microbatch_size=1,
        generation_batch_size=1,
        validation_samples=2,
        n_layers=2,
        evaluation_batch_size=7,
        minimum_in_range_samples=1,
        minimum_in_range_per_pilot=1,
    )
    assert captured["evaluation_batch_size"] == 7


def test_sensitivity_decisions_require_both_screening_and_confirmation_seeds():
    records = []
    for seed, metric, improves in (
        (2026, 0.20, True),
        (2027, 0.21, True),
        (2028, 0.22, True),
    ):
        records.append(
            {
                "variable": "time_mixer_channels",
                "value": 4,
                "seed": seed,
                "phase": "confirmation" if seed == 2028 else "screening",
                "status": "complete",
                "finite": True,
                "training_steps": 2000,
                "validation_mean_eqdeeprx_ber": metric,
                "baseline_validation_mean_eqdeeprx_ber": 0.30,
                "improves_baseline": improves,
            }
        )
    decisions = summarize_candidate_decisions(
        records,
        screening_seeds=(2026, 2027),
        confirmation_seed=2028,
    )
    assert decisions[0]["decision"] == "confirmed_improvement"
    assert decisions[0]["screening_improvements"] == [2026, 2027]
    assert decisions[0]["confirmation_improvement"] is True


def test_short_confirmation_is_not_marked_as_confirmed_improvement():
    records = []
    for seed in (2026, 2027, 2028):
        records.append(
            {
                "variable": "time_mixer_channels",
                "value": 4,
                "seed": seed,
                "phase": "confirmation" if seed == 2028 else "screening",
                "status": "complete",
                "finite": True,
                "training_steps": 20,
                "validation_mean_eqdeeprx_ber": 0.20,
                "baseline_validation_mean_eqdeeprx_ber": 0.30,
                "improves_baseline": True,
            }
        )
    decisions = summarize_candidate_decisions(
        records,
        screening_seeds=(2026, 2027),
        confirmation_seed=2028,
    )
    assert decisions[0]["decision"] == "screening_improvement_only"
    assert decisions[0]["confirmation_improvement"] is False
    assert any("confirmation steps" in item for item in decisions[0]["rejection_reasons"])


def test_sensitivity_decisions_record_missing_or_nonfinite_evidence():
    decisions = summarize_candidate_decisions(
        [
            {
                "variable": "amp_dtype",
                "value": "float32",
                "seed": 2026,
                "phase": "screening",
                "status": "error",
                "finite": False,
                "improves_baseline": False,
            }
        ],
        screening_seeds=(2026, 2027),
        confirmation_seed=2028,
    )
    assert decisions[0]["decision"] == "rejected"
    assert "missing screening seed 2027" in decisions[0]["rejection_reasons"]
    assert "nonfinite or incomplete seed 2026" in decisions[0]["rejection_reasons"]


def test_sensitivity_summary_contains_machine_readable_candidate_decisions():
    records = [
        {
            "variable": "amp_dtype",
            "value": "bfloat16",
            "seed": seed,
            "status": "complete",
            "finite": True,
            "validation_mean_eqdeeprx_ber": 0.3,
            "baseline_value": "bfloat16",
            "improves_baseline": False,
        }
        for seed in (2026, 2027, 2028)
    ]
    summary = build_sensitivity_summary(
        variables=("amp_dtype",),
        matrix={"amp_dtype": ["bfloat16"]},
        screening_seeds=(2026, 2027),
        confirmation_seed=2028,
        coverage_policy={"minimum_in_range_samples": 100},
        records=records,
    )
    assert summary["candidate_decisions"][0]["decision"] == "baseline"
    assert summary["candidate_decisions"][0]["seeds_present"] == [2026, 2027, 2028]
