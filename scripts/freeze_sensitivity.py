from __future__ import annotations

"""Validate and freeze the unpublished-configuration sensitivity evidence.

The sensitivity jobs are exploratory and never write formal checkpoints.  This
tool turns their independent JSON summaries into an auditable gate for the
formal run.  It deliberately distinguishes a finite, preregistered candidate
winner from a mathematical global optimum: the former can be frozen after the
joint profile check, while the latter is not claimed from a finite experiment.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.run_unpublished_sensitivity import DEFAULT_CANDIDATES

SCREENING_SEEDS = (2026, 2027)
CONFIRMATION_SEED = 2028
MIN_CONFIRMATION_STEPS = 2_000
SCHEMA_VERSION = 5
EXPECTED_VARIABLES = (
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
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def load_joint_summary(path: Path) -> dict[str, Any]:
    """Read a joint-validation summary for the final freeze gate."""

    return _read_json(Path(path))


def _joint_summary_is_valid(joint_summary: dict[str, Any] | None) -> bool:
    """Require complete provenance and finite evidence for every joint record."""

    if not isinstance(joint_summary, dict):
        return False
    if not (
        joint_summary.get("status") == "complete"
        and joint_summary.get("valid") is True
        and not joint_summary.get("missing_records")
        and joint_summary.get("formal_checkpoint_touched") is False
    ):
        return False
    records = joint_summary.get("records")
    if not isinstance(records, list) or not records:
        return False
    try:
        minimum_steps = int(joint_summary.get("minimum_confirmation_steps", MIN_CONFIRMATION_STEPS))
    except (TypeError, ValueError):
        return False
    if minimum_steps < MIN_CONFIRMATION_STEPS:
        return False
    profiles = joint_summary.get("profiles")
    seeds = joint_summary.get("seeds")
    if not isinstance(profiles, list) or not profiles:
        return False
    if "formal_baseline" not in profiles:
        return False
    if not isinstance(seeds, list) or not seeds:
        return False
    try:
        expected_pairs = {
            (str(profile), int(seed)) for profile in profiles for seed in seeds
        }
    except (TypeError, ValueError):
        return False
    if len(records) != len(expected_pairs):
        return False
    declared_count = joint_summary.get("record_count")
    if declared_count is not None:
        try:
            if int(declared_count) != len(records):
                return False
        except (TypeError, ValueError):
            return False
    observed_pairs: set[tuple[str, int]] = set()
    metrics_by_profile: dict[str, dict[int, float]] = {}
    for record in records:
        if not isinstance(record, dict):
            return False
        try:
            pair = (str(record["profile_name"]), int(record["seed"]))
        except (KeyError, TypeError, ValueError):
            return False
        if pair in observed_pairs or pair not in expected_pairs:
            return False
        observed_pairs.add(pair)
        if not (
            record.get("status") == "complete"
            and record.get("finite") is True
            and record.get("formal_checkpoint_touched") is False
            and record.get("validation_coverage_ok") is True
        ):
            return False
        metric = record.get("validation_mean_eqdeeprx_ber")
        if not isinstance(metric, (int, float)) or not math.isfinite(float(metric)):
            return False
        metrics_by_profile.setdefault(pair[0], {})[pair[1]] = float(metric)
        try:
            if int(record.get("training_steps", -1)) < minimum_steps:
                return False
        except (TypeError, ValueError):
            return False
    if observed_pairs != expected_pairs:
        return False
    baseline_metrics = metrics_by_profile.get("formal_baseline")
    if baseline_metrics is None or set(baseline_metrics) != set(int(seed) for seed in seeds):
        return False
    eligible_profiles = ["formal_baseline"]
    for profile in profiles:
        if profile == "formal_baseline":
            continue
        profile_metrics = metrics_by_profile.get(profile, {})
        if all(
            seed in profile_metrics
            and profile_metrics[seed] < baseline_metrics[seed]
            for seed in (int(item) for item in seeds)
        ):
            eligible_profiles.append(str(profile))
    best_profile = joint_summary.get("best_profile")
    if best_profile not in eligible_profiles:
        return False
    declared_eligible = joint_summary.get("eligible_profiles")
    if declared_eligible is not None and sorted(str(item) for item in declared_eligible) != sorted(eligible_profiles):
        return False
    return True


def load_summaries(root: Path) -> list[dict[str, Any]]:
    """Load one sensitivity summary per variable directory in stable order."""

    root = Path(root)
    summaries: list[dict[str, Any]] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        path = directory / "sensitivity_summary.json"
        if path.is_file():
            summary = _read_json(path)
            summary["_source_path"] = str(path)
            summaries.append(summary)
    return summaries


def _record_key(record: dict[str, Any]) -> tuple[str, str]:
    return str(record.get("variable")), repr(record.get("value"))


def _value_key(value: Any) -> str:
    """Canonicalize JSON scalar values for matrix/record identity checks."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_confirmation_matrix(
    summaries: Iterable[dict[str, Any]],
    *,
    screening_seeds: tuple[int, ...] = SCREENING_SEEDS,
    confirmation_seed: int = CONFIRMATION_SEED,
    minimum_confirmation_steps: int = MIN_CONFIRMATION_STEPS,
    expected_variables: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Validate every summary and every seed record before freezing values."""

    errors: list[str] = []
    summary_list = list(summaries)
    expected_seeds = tuple(int(seed) for seed in (*screening_seeds, confirmation_seed))
    all_records: list[dict[str, Any]] = []
    requested_variables = tuple(
        str(variable)
        for variable in (expected_variables if expected_variables is not None else EXPECTED_VARIABLES)
    )
    require_full_matrix = set(requested_variables) == set(EXPECTED_VARIABLES)
    observed_matrix: dict[str, set[str]] = {}

    if not summary_list:
        errors.append("no sensitivity summaries found")

    if expected_variables is not None:
        observed_variables = {
            str(decision.get("variable"))
            for summary in summary_list
            for decision in summary.get("candidate_decisions", [])
            if isinstance(decision, dict)
        }
        missing_variables = sorted(set(expected_variables) - observed_variables)
        if missing_variables:
            errors.append(f"missing sensitivity variables: {missing_variables}")

    for summary in summary_list:
        source = summary.get("_source_path", "<summary>")
        if summary.get("sensitivity_schema_version") not in {None, SCHEMA_VERSION}:
            errors.append(f"{source}: unsupported schema version")
        if summary.get("status") != "complete" or summary.get("execution_complete") is not True:
            errors.append(f"{source}: summary is incomplete")
        if summary.get("missing_records"):
            errors.append(f"{source}: summary has missing records")
        records = summary.get("records")
        if not isinstance(records, list) or not records:
            errors.append(f"{source}: records are missing")
            continue
        all_records.extend(record for record in records if isinstance(record, dict))
        if require_full_matrix:
            matrix = summary.get("candidate_matrix")
            if not isinstance(matrix, dict):
                errors.append(f"{source}: candidate_matrix is missing")
            else:
                for variable, values in matrix.items():
                    if variable in requested_variables and isinstance(values, list):
                        observed_matrix.setdefault(str(variable), set()).update(
                            _value_key(value) for value in values
                        )
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for record in records:
            if not isinstance(record, dict):
                errors.append(f"{source}: non-object record")
                continue
            grouped.setdefault(_record_key(record), []).append(record)
            if record.get("status") != "complete":
                errors.append(f"{source}: incomplete record {_record_key(record)}")
            if record.get("finite") is not True:
                errors.append(f"{source}: non-finite record {_record_key(record)}")
            if record.get("formal_checkpoint_touched") is not False:
                errors.append(f"{source}: formal checkpoint touched {_record_key(record)}")
            if record.get("validation_coverage_ok") is not True:
                errors.append(f"{source}: validation coverage failed {_record_key(record)}")
            metric = record.get("validation_mean_eqdeeprx_ber")
            if not isinstance(metric, (int, float)) or not math.isfinite(float(metric)):
                errors.append(f"{source}: invalid BER metric {_record_key(record)}")
            try:
                seed = int(record["seed"])
                steps = int(record.get("training_steps", record.get("screening_steps")))
            except (KeyError, TypeError, ValueError):
                errors.append(f"{source}: invalid seed/step fields {_record_key(record)}")
                continue
            if seed not in expected_seeds:
                errors.append(f"{source}: unexpected seed {seed}")
            if steps < int(minimum_confirmation_steps):
                errors.append(
                    f"{source}: seed {seed} has {steps} steps; "
                    f"requires at least {int(minimum_confirmation_steps)}"
                )

        for key, records_for_value in grouped.items():
            seeds = sorted(int(item.get("seed")) for item in records_for_value if "seed" in item)
            if seeds != sorted(expected_seeds):
                errors.append(
                    f"{source}: {_record_key(records_for_value[0])} has seeds {seeds}; "
                    f"expected {sorted(expected_seeds)}"
                )
            step_values = {
                int(item.get("training_steps", item.get("screening_steps")))
                for item in records_for_value
                if item.get("training_steps", item.get("screening_steps")) is not None
            }
            if len(step_values) > 1:
                errors.append(
                    f"{source}: {_record_key(records_for_value[0])} has unequal "
                    f"seed budgets {sorted(step_values)}"
                )

    if require_full_matrix:
        for variable in requested_variables:
            expected_values = {
                _value_key(value) for value in DEFAULT_CANDIDATES[variable]
            }
            actual_values = observed_matrix.get(variable, set())
            if actual_values != expected_values:
                errors.append(
                    f"candidate matrix for {variable} does not match the registered values"
                )
        expected_keys = {
            (variable, _value_key(value), int(seed))
            for variable in requested_variables
            for value in DEFAULT_CANDIDATES[variable]
            for seed in expected_seeds
        }
        observed_keys: set[tuple[str, str, int]] = set()
        for record in all_records:
            try:
                key = (
                    str(record["variable"]),
                    _value_key(record["value"]),
                    int(record["seed"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            if key in observed_keys:
                errors.append(f"duplicate sensitivity record: {key}")
            observed_keys.add(key)
        missing_keys = sorted(expected_keys - observed_keys)
        unexpected_keys = sorted(observed_keys - expected_keys)
        if missing_keys:
            errors.append(
                f"missing full-matrix records: {len(missing_keys)} (expected {len(expected_keys)})"
            )
        if unexpected_keys:
            errors.append(f"unexpected full-matrix records: {len(unexpected_keys)}")

    return {
        "ok": not errors,
        "errors": errors,
        "summary_count": len(summary_list),
        "record_count": len(all_records),
        "expected_record_count": (
            sum(len(DEFAULT_CANDIDATES[variable]) for variable in requested_variables)
            * len(expected_seeds)
            if require_full_matrix
            else None
        ),
        "seeds": list(expected_seeds),
        "minimum_confirmation_steps": int(minimum_confirmation_steps),
    }


def _decision_metric(decision: dict[str, Any]) -> float:
    metrics = decision.get("metrics_by_seed", {})
    values = [float(metrics[str(seed)]) for seed in (*SCREENING_SEEDS, CONFIRMATION_SEED) if str(seed) in metrics]
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        return math.inf
    return sum(values) / len(values)


def _flatten_decisions(summaries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for summary in summaries:
        for decision in summary.get("candidate_decisions", []):
            if isinstance(decision, dict):
                decisions.append(dict(decision))
    return decisions


def build_freeze_manifest(
    root: Path,
    *,
    joint_summary: dict[str, Any] | None = None,
    expected_variables: Iterable[str] | None = EXPECTED_VARIABLES,
) -> dict[str, Any]:
    """Build the machine-readable pre-training configuration gate."""

    summaries = load_summaries(Path(root))
    validation = validate_confirmation_matrix(
        summaries,
        expected_variables=expected_variables,
    )
    decisions = _flatten_decisions(summaries)
    baselines: dict[str, Any] = {}
    confirmed_by_variable: dict[str, list[dict[str, Any]]] = {}
    for decision in decisions:
        variable = str(decision.get("variable"))
        if decision.get("decision") == "baseline":
            baselines[variable] = decision.get("value")
        if decision.get("decision") == "confirmed_improvement":
            confirmed_by_variable.setdefault(variable, []).append(decision)

    selected: dict[str, Any] = {}
    confirmed_candidates: list[dict[str, Any]] = []
    for variable, candidates in sorted(confirmed_by_variable.items()):
        ranked = sorted(candidates, key=lambda item: (_decision_metric(item), repr(item.get("value"))))
        winner = ranked[0]
        selected[variable] = winner.get("value")
        confirmed_candidates.extend(
            {
                "variable": variable,
                "value": item.get("value"),
                "decision": item.get("decision"),
                "mean_confirmation_metric": _decision_metric(item),
                "metrics_by_seed": item.get("metrics_by_seed", {}),
            }
            for item in ranked
        )

    joint_ok = _joint_summary_is_valid(joint_summary)
    has_exploratory_overrides = bool(selected)
    if not validation["ok"]:
        status = "blocked"
    elif has_exploratory_overrides and not joint_ok:
        status = "needs_joint_validation"
    else:
        status = "ready_for_long_training"

    joint_profile_name = joint_summary.get("best_profile") if joint_summary else None
    joint_profile_overrides = {}
    if joint_ok and joint_profile_name:
        joint_profile_overrides = dict(
            joint_summary.get("profile_overrides", {}).get(joint_profile_name, {})
        )

    return {
        "freeze_schema_version": 1,
        "sensitivity_schema_version": SCHEMA_VERSION,
        "status": status,
        "root": str(Path(root)),
        "validation": validation,
        "formal_reproduction_profile": baselines,
        "best_confirmed_exploratory_profile": selected,
        "confirmed_candidates": confirmed_candidates,
        "joint_validation_required": bool(has_exploratory_overrides and not joint_ok),
        "joint_validation": joint_summary,
        "joint_selected_profile": joint_profile_name if joint_ok else None,
        "joint_selected_overrides": joint_profile_overrides,
        "launch_profile": (
            joint_profile_overrides if joint_ok else baselines
        ),
        "global_optimum_claim": False,
        "scope": {
            "description": "best validated choice within the preregistered finite candidate matrix",
            "screening_seeds": list(SCREENING_SEEDS),
            "confirmation_seed": CONFIRMATION_SEED,
            "one_variable_screening": True,
            "confirmation_steps": MIN_CONFIRMATION_STEPS,
            "paper_private_random_stream_unavailable": True,
        },
        "figure6a_protocol": {
            "n_layers": 4,
            "denoisenet_only": False,
            "curves": [
                "eqdeeprx_1_pilot",
                "eqdeeprx_2_pilots",
                "baseline_1_pilot",
                "baseline_2_pilots",
                "baseline_known_channel",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--joint-summary",
        type=Path,
        help="completed joint-validation JSON produced by run_joint_sensitivity.py",
    )
    args = parser.parse_args()
    joint_summary = load_joint_summary(args.joint_summary) if args.joint_summary else None
    manifest = build_freeze_manifest(args.root, joint_summary=joint_summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, allow_nan=False))
    if manifest["status"] != "ready_for_long_training":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
