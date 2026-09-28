import json
from pathlib import Path

from eqdeeprx.config import paper_config
from scripts.run_joint_sensitivity import (
    JOINT_SCHEMA_VERSION,
    apply_profile,
    build_joint_summary,
    load_matching_record,
    profile_fingerprint,
    reached_record_limit,
)


def _record(profile, seed, *, profile_name="formal_baseline", steps=2000, metric=0.3):
    return {
        "joint_schema_version": JOINT_SCHEMA_VERSION,
        "profile": profile,
        "profile_name": profile_name,
        "seed": seed,
        "status": "complete",
        "finite": True,
        "formal_checkpoint_touched": False,
        "validation_coverage_ok": True,
        "training_steps": steps,
        "validation_mean_eqdeeprx_ber": metric,
    }


def test_apply_profile_changes_only_declared_unpublished_values():
    base = paper_config()
    candidate = apply_profile(
        base,
        {"cdl_delay_spread_mode": "fixed_100ns", "lamb_bias_correction": False},
    )
    assert candidate.channel.cdl_delay_spread_mode == "fixed"
    assert candidate.cdl_delay_spread_ns == 100.0
    assert candidate.training.lamb_bias_correction is False
    assert candidate.model.time_mixer_channels == base.model.time_mixer_channels


def test_joint_summary_blocks_missing_or_short_profiles():
    records = [_record({}, 2026), _record({}, 2027), _record({}, 2028, steps=20)]
    summary = build_joint_summary(
        profiles=["formal_baseline"], records=records, seeds=(2026, 2027, 2028)
    )
    assert summary["status"] == "incomplete"
    assert summary["missing_records"]


def test_joint_summary_is_complete_only_for_all_finite_profiles():
    records = []
    for profile in ({}, {"cdl_delay_spread_mode": "fixed_100ns"}):
        profile_name = "formal_baseline" if not profile else "fixed100"
        for seed in (2026, 2027, 2028):
            records.append(
                _record(
                    profile,
                    seed,
                    profile_name=profile_name,
                    metric=0.2 if profile else 0.3,
                )
            )
    summary = build_joint_summary(
        profiles=["formal_baseline", "fixed100"],
        records=records,
        seeds=(2026, 2027, 2028),
    )
    assert summary["status"] == "complete"
    assert summary["missing_records"] == []
    assert summary["best_profile"] == "fixed100"


def test_joint_resume_rejects_a_record_with_changed_profile(tmp_path: Path):
    path = tmp_path / "record.json"
    record = _record(
        {"cdl_delay_spread_mode": "fixed_100ns"},
        2026,
        profile_name="fixed100",
    )
    path.write_text(json.dumps(record), encoding="utf-8")
    assert (
        load_matching_record(
            path,
            profile_name="fixed100",
            profile={"cdl_delay_spread_mode": "fixed_300ns"},
            seed=2026,
            steps=2000,
        )
        is None
    )


def test_joint_resume_rejects_a_record_with_changed_protocol(tmp_path: Path):
    profile = {}
    record = _record(profile, 2026)
    record.update(
        {
            "profile_fingerprint": profile_fingerprint(profile),
            "backend": "fast",
            "config_fingerprint": "config-a",
            "validation_samples": 400,
            "batch_size": 8,
            "microbatch_size": 4,
            "generation_batch_size": 2,
            "n_layers": 4,
            "evaluation_batch_size": 2,
            "minimum_in_range_samples": 100,
            "minimum_in_range_per_pilot": 20,
            "system_seed": 123,
            "validation_plan_fingerprint": "plan-a",
        }
    )
    path = tmp_path / "record.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    assert (
        load_matching_record(
            path,
            profile_name="formal_baseline",
            profile=profile,
            seed=2026,
            steps=2000,
            backend="sionna",
            expected_config_fingerprint="config-a",
            validation_samples=400,
            batch_size=8,
            microbatch_size=4,
            generation_batch_size=2,
            n_layers=4,
            evaluation_batch_size=2,
            minimum_in_range_samples=100,
            minimum_in_range_per_pilot=20,
            expected_system_seed=123,
            expected_validation_plan_fingerprint="plan-a",
        )
        is None
    )


def test_joint_summary_rejects_candidate_that_regresses_one_seed():
    records = []
    for seed in (2026, 2027, 2028):
        records.append(_record({}, seed, metric=0.3))
        records.append(
            _record(
                {"cdl_delay_spread_mode": "fixed_100ns"},
                seed,
                profile_name="fixed100",
                metric={2026: 0.2, 2027: 0.4, 2028: 0.2}[seed],
            )
        )
    summary = build_joint_summary(
        profiles=["formal_baseline", "fixed100"],
        records=records,
        seeds=(2026, 2027, 2028),
        profile_overrides={"formal_baseline": {}, "fixed100": {"cdl_delay_spread_mode": "fixed_100ns"}},
    )
    assert summary["status"] == "complete"
    assert summary["best_profile"] == "formal_baseline"
    assert "fixed100" not in summary["eligible_profiles"]


def test_joint_record_limit_is_optional_and_inclusive():
    assert reached_record_limit(10, None) is False
    assert reached_record_limit(0, 1) is False
    assert reached_record_limit(1, 1) is True
    assert reached_record_limit(2, 1) is True


def test_joint_record_limit_counts_only_completed_records_in_runner_contract():
    records = [
        {"status": "error"},
        {"status": "complete"},
    ]
    completed = sum(record.get("status") == "complete" for record in records)
    assert reached_record_limit(completed, 2) is False
    assert reached_record_limit(completed, 1) is True
