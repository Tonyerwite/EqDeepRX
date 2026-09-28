import json
from pathlib import Path

from scripts.freeze_sensitivity import (
    _joint_summary_is_valid,
    build_freeze_manifest,
    load_joint_summary,
    validate_confirmation_matrix,
)
from scripts.run_unpublished_sensitivity import DEFAULT_CANDIDATES


def _record(value, seed, *, phase, steps=2000, metric=0.4, baseline=0.5):
    return {
        "variable": "example",
        "value": value,
        "seed": seed,
        "phase": phase,
        "status": "complete",
        "finite": True,
        "formal_checkpoint_touched": False,
        "validation_coverage_ok": True,
        "training_steps": steps,
        "validation_mean_eqdeeprx_ber": metric,
        "baseline_validation_mean_eqdeeprx_ber": baseline,
        "improves_baseline": metric < baseline,
    }


def _summary(value, decision="baseline", *, include_confirmation=True):
    records = []
    for seed in (2026, 2027):
        records.append(_record(value, seed, phase="screening", metric=0.4))
    if include_confirmation:
        records.append(_record(value, 2028, phase="confirmation", steps=2000, metric=0.4))
    return {
        "sensitivity_schema_version": 5,
        "status": "complete",
        "execution_complete": True,
        "missing_records": [],
        "records": records,
        "candidate_decisions": [
            {
                "variable": "example",
                "value": value,
                "decision": decision,
                "screening_improvements": [2026, 2027] if decision == "confirmed_improvement" else [],
                "confirmation_improvement": decision == "confirmed_improvement",
                "seeds_present": [2026, 2027, 2028] if include_confirmation else [2026, 2027],
                "metrics_by_seed": {"2026": 0.4, "2027": 0.4, "2028": 0.4},
            }
        ],
    }


def test_confirmation_matrix_rejects_missing_confirmation_record():
    summary = _summary("baseline", include_confirmation=False)
    result = validate_confirmation_matrix([summary])
    assert result["ok"] is False
    assert any("2028" in error for error in result["errors"])


def test_confirmation_matrix_rejects_unequal_seed_budgets():
    summary = _summary("baseline")
    summary["records"][0]["training_steps"] = 20
    result = validate_confirmation_matrix([summary])
    assert result["ok"] is False
    assert any("requires at least" in error for error in result["errors"])
    assert any("unequal seed budgets" in error for error in result["errors"])


def test_full_matrix_gate_rejects_a_baseline_only_summary():
    summary = _summary("baseline")
    summary["candidate_matrix"] = {
        variable: list(values) for variable, values in DEFAULT_CANDIDATES.items()
    }
    result = validate_confirmation_matrix([summary])
    assert result["ok"] is False
    assert any("missing full-matrix records" in error for error in result["errors"])


def test_freeze_manifest_requires_joint_evidence_for_multiple_overrides(tmp_path: Path):
    baseline = _summary("baseline", decision="baseline")
    candidate = _summary("candidate", decision="confirmed_improvement")
    # Give the synthetic candidate its own variable identity so the fixture
    # exercises the multi-variable profile rule.
    candidate["candidate_decisions"][0]["variable"] = "other"
    for record in candidate["records"]:
        record["variable"] = "other"
    root = tmp_path / "confirmation"
    for name, payload in (("example", baseline), ("other", candidate)):
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "sensitivity_summary.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    manifest = build_freeze_manifest(root, expected_variables=("example", "other"))
    assert manifest["status"] == "needs_joint_validation"
    assert manifest["formal_reproduction_profile"]["example"] == "baseline"
    assert manifest["best_confirmed_exploratory_profile"]["other"] == "candidate"
    assert manifest["global_optimum_claim"] is False


def test_freeze_manifest_is_ready_when_only_baselines_are_retained(tmp_path: Path):
    root = tmp_path / "confirmation"
    directory = root / "example"
    directory.mkdir(parents=True)
    (directory / "sensitivity_summary.json").write_text(
        json.dumps(_summary("baseline")), encoding="utf-8"
    )
    manifest = build_freeze_manifest(root, expected_variables=("example",))
    assert manifest["status"] == "ready_for_long_training"
    assert manifest["joint_validation_required"] is False
    assert manifest["figure6a_protocol"]["denoisenet_only"] is False


def test_freeze_manifest_accepts_a_valid_joint_profile(tmp_path: Path):
    baseline = _summary("baseline", decision="baseline")
    candidate = _summary("candidate", decision="confirmed_improvement")
    candidate["candidate_decisions"][0]["variable"] = "other"
    for record in candidate["records"]:
        record["variable"] = "other"
    root = tmp_path / "confirmation"
    for name, payload in (("example", baseline), ("other", candidate)):
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "sensitivity_summary.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    joint = {
        "joint_schema_version": 1,
        "status": "complete",
        "valid": True,
        "missing_records": [],
        "formal_checkpoint_touched": False,
        "minimum_confirmation_steps": 2000,
        "profiles": ["formal_baseline", "joint"],
        "seeds": [2026, 2027, 2028],
        "best_profile": "joint",
        "eligible_profiles": ["formal_baseline", "joint"],
        "profile_overrides": {
            "formal_baseline": {},
            "joint": {"other": "candidate"},
        },
        "records": [
            {
                "profile_name": profile_name,
                "seed": seed,
                "status": "complete",
                "finite": True,
                "formal_checkpoint_touched": False,
                "validation_coverage_ok": True,
                "training_steps": 2000,
                "validation_mean_eqdeeprx_ber": 0.3 if profile_name == "formal_baseline" else 0.2,
            }
            for profile_name in ("formal_baseline", "joint")
            for seed in (2026, 2027, 2028)
        ],
    }
    manifest = build_freeze_manifest(
        root,
        joint_summary=joint,
        expected_variables=("example", "other"),
    )
    assert manifest["status"] == "ready_for_long_training"
    assert manifest["joint_validation_required"] is False
    assert manifest["joint_selected_profile"] == "joint"
    assert manifest["launch_profile"] == {"other": "candidate"}


def test_joint_summary_loader_reads_a_machine_readable_summary(tmp_path: Path):
    path = tmp_path / "joint_summary.json"
    payload = {"status": "complete", "records": []}
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_joint_summary(path) == payload


def test_freeze_manifest_rejects_an_invalid_joint_record(tmp_path: Path):
    baseline = _summary("baseline", decision="baseline")
    candidate = _summary("candidate", decision="confirmed_improvement")
    candidate["candidate_decisions"][0]["variable"] = "other"
    for record in candidate["records"]:
        record["variable"] = "other"
    root = tmp_path / "confirmation"
    for name, payload in (("example", baseline), ("other", candidate)):
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "sensitivity_summary.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    joint = {
        "status": "complete",
        "valid": True,
        "missing_records": [],
        "formal_checkpoint_touched": False,
        "minimum_confirmation_steps": 2000,
        "records": [
            {
                "status": "complete",
                "finite": False,
                "formal_checkpoint_touched": False,
                "validation_coverage_ok": True,
                "training_steps": 2000,
                "validation_mean_eqdeeprx_ber": 0.2,
            }
        ],
    }
    manifest = build_freeze_manifest(
        root,
        joint_summary=joint,
        expected_variables=("example", "other"),
    )
    assert manifest["status"] == "needs_joint_validation"


def test_joint_summary_requires_each_profile_seed_pair():
    summary = {
        "status": "complete",
        "valid": True,
        "missing_records": [],
        "formal_checkpoint_touched": False,
        "minimum_confirmation_steps": 2000,
        "profiles": ["joint"],
        "seeds": [2026, 2027, 2028],
        "record_count": 2,
        "records": [
            {
                "profile_name": "joint",
                "seed": seed,
                "status": "complete",
                "finite": True,
                "formal_checkpoint_touched": False,
                "validation_coverage_ok": True,
                "training_steps": 2000,
                "validation_mean_eqdeeprx_ber": 0.2,
            }
            for seed in (2026, 2027)
        ],
    }
    assert _joint_summary_is_valid(summary) is False


def test_joint_summary_rejects_profile_that_regresses_against_formal_baseline():
    records = []
    for seed in (2026, 2027, 2028):
        records.append(
            {
                "profile_name": "formal_baseline",
                "seed": seed,
                "status": "complete",
                "finite": True,
                "formal_checkpoint_touched": False,
                "validation_coverage_ok": True,
                "training_steps": 2000,
                "validation_mean_eqdeeprx_ber": 0.3,
            }
        )
        records.append(
            {
                "profile_name": "candidate",
                "seed": seed,
                "status": "complete",
                "finite": True,
                "formal_checkpoint_touched": False,
                "validation_coverage_ok": True,
                "training_steps": 2000,
                "validation_mean_eqdeeprx_ber": {2026: 0.2, 2027: 0.4, 2028: 0.2}[seed],
            }
        )
    summary = {
        "status": "complete",
        "valid": True,
        "missing_records": [],
        "formal_checkpoint_touched": False,
        "minimum_confirmation_steps": 2000,
        "profiles": ["formal_baseline", "candidate"],
        "seeds": [2026, 2027, 2028],
        "record_count": len(records),
        "best_profile": "candidate",
        "records": records,
    }
    assert _joint_summary_is_valid(summary) is False
