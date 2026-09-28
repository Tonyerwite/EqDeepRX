from __future__ import annotations

"""Validate combinations of confirmed unpublished choices.

This runner is intentionally separate from the one-variable screening
protocol.  It reuses the same Sionna backend, seed-derived validation plan,
four-layer Figure 6(a) protocol, and no-formal-checkpoint rule, but evaluates
the small set of profiles that survived the preregistered confirmation gate.
"""

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from eqdeeprx.config import EqDeepRxConfig, paper_config
from eqdeeprx.evaluation import _model_fingerprint
from eqdeeprx.model import EqDeepRx
from eqdeeprx.training import config_fingerprint, train_steps
from scripts.run_unpublished_sensitivity import (
    MIN_CONFIRMATION_STEPS,
    _apply_reference_state,
    _candidate_system_seed,
    _evaluate,
    _mean_eqdeeprx_ber,
    _reference_model_state,
    _reset_system_rng,
    apply_candidate,
    assess_validation_coverage,
    validation_plan_fingerprint,
)

JOINT_SCHEMA_VERSION = 1

# These are the only combinations admitted by the preceding confirmation
# stage.  The fixed delay-spread variants remain exploratory choices; they are
# not asserted to be the paper's published validation distribution.
JOINT_PROFILES: dict[str, dict[str, Any]] = {
    "formal_baseline": {},
    "fixed100": {"cdl_delay_spread_mode": "fixed_100ns"},
    "fixed300": {"cdl_delay_spread_mode": "fixed_300ns"},
    "bias_no_correction": {"lamb_bias_correction": False},
    "fixed100_bias_no_correction": {
        "cdl_delay_spread_mode": "fixed_100ns",
        "lamb_bias_correction": False,
    },
    "fixed300_bias_no_correction": {
        "cdl_delay_spread_mode": "fixed_300ns",
        "lamb_bias_correction": False,
    },
}


def apply_profile(config: EqDeepRxConfig, profile: dict[str, Any]) -> EqDeepRxConfig:
    """Apply an explicit multi-variable profile in deterministic key order."""

    result = config
    for variable in sorted(profile):
        result = apply_candidate(result, variable, profile[variable])
    return result


def profile_fingerprint(profile: dict[str, Any]) -> str:
    payload = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def reached_record_limit(record_count: int, max_records: int | None) -> bool:
    """Return whether an optional clean-stop limit has been reached."""

    return max_records is not None and int(record_count) >= int(max_records)


def _finite_record(record: dict[str, Any], minimum_steps: int) -> bool:
    metric = record.get("validation_mean_eqdeeprx_ber")
    return bool(
        record.get("status") == "complete"
        and record.get("finite") is True
        and record.get("formal_checkpoint_touched") is False
        and record.get("validation_coverage_ok") is True
        and int(record.get("training_steps", 0)) >= int(minimum_steps)
        and isinstance(metric, (int, float))
        and math.isfinite(float(metric))
    )


def build_joint_summary(
    *,
    profiles: list[str] | tuple[str, ...],
    records: list[dict[str, Any]],
    seeds: tuple[int, ...] = (2026, 2027, 2028),
    minimum_steps: int = MIN_CONFIRMATION_STEPS,
    profile_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a complete/incomplete summary and choose the best tested profile."""

    profile_names = list(profiles)
    missing: list[dict[str, Any]] = []
    profile_metrics: dict[str, dict[str, float]] = {}
    for name in profile_names:
        items = [record for record in records if record.get("profile_name") == name]
        by_seed = {int(record.get("seed")): record for record in items if "seed" in record}
        for seed in seeds:
            record = by_seed.get(int(seed))
            if record is None:
                missing.append({"profile": name, "seed": int(seed), "reason": "missing"})
                continue
            if not _finite_record(record, minimum_steps):
                missing.append({"profile": name, "seed": int(seed), "reason": "invalid"})
        if len(by_seed) == len(seeds) and all(
            _finite_record(by_seed[int(seed)], minimum_steps) for seed in seeds
        ):
            profile_metrics[name] = {
                str(seed): float(by_seed[int(seed)]["validation_mean_eqdeeprx_ber"])
                for seed in seeds
            }

    baseline_metrics = profile_metrics.get("formal_baseline")
    profile_improvements: dict[str, dict[str, bool]] = {}
    eligible_profiles: list[str] = []
    if baseline_metrics is not None:
        for name, metrics in profile_metrics.items():
            improves = {
                str(seed): (
                    name == "formal_baseline"
                    or float(metrics[str(seed)]) < float(baseline_metrics[str(seed)])
                )
                for seed in seeds
            }
            profile_improvements[name] = improves
            if all(improves.values()):
                eligible_profiles.append(name)

    best_profile = None
    if not missing and profile_metrics:
        # Always retain the formal baseline as a safe fallback. Exploratory
        # profiles are eligible only when every seed beats that same-seed
        # baseline; an average improvement with a regressing seed is rejected.
        ranked_profiles = [
            name for name in eligible_profiles if name in profile_metrics
        ]
        if ranked_profiles:
            best_profile = min(
                ranked_profiles,
                key=lambda name: (
                    sum(profile_metrics[name].values()) / len(profile_metrics[name]),
                    max(profile_metrics[name].values()),
                    name,
                ),
            )
    return {
        "joint_schema_version": JOINT_SCHEMA_VERSION,
        "status": "complete" if not missing else "incomplete",
        "valid": not missing,
        "profiles": profile_names,
        "profile_overrides": profile_overrides or {},
        "seeds": [int(seed) for seed in seeds],
        "minimum_confirmation_steps": int(minimum_steps),
        "missing_records": missing,
        "record_count": len(records),
        "profile_metrics": profile_metrics,
        "baseline_metrics": baseline_metrics,
        "profile_improvements": profile_improvements,
        "eligible_profiles": eligible_profiles,
        "best_profile": best_profile,
        "formal_checkpoint_touched": any(
            record.get("formal_checkpoint_touched") is not False for record in records
        ),
        "records": records,
    }


def run_profile(
    config: EqDeepRxConfig,
    *,
    profile_name: str,
    profile: dict[str, Any],
    seed: int,
    backend: str,
    device: str,
    training_steps: int,
    batch_size: int,
    microbatch_size: int,
    generation_batch_size: int,
    validation_samples: int,
    n_layers: int,
    evaluation_batch_size: int,
    minimum_in_range_samples: int,
    minimum_in_range_per_pilot: int,
    reference_state: dict[str, torch.Tensor] | None,
    reference_initialization_fingerprint: str | None,
) -> dict[str, Any]:
    profile_config = apply_profile(config, profile)
    candidate = replace(
        profile_config,
        training=replace(profile_config.training, seed=seed),
    )
    system_seed = _candidate_system_seed(seed)
    record: dict[str, Any] = {
        "joint_schema_version": JOINT_SCHEMA_VERSION,
        "profile_name": profile_name,
        "profile": profile,
        "seed": int(seed),
        "config_fingerprint": config_fingerprint(candidate),
        "profile_fingerprint": profile_fingerprint(profile),
        "backend": backend,
        "training_steps": int(training_steps),
        "validation_samples": int(validation_samples),
        "batch_size": int(batch_size),
        "microbatch_size": int(microbatch_size),
        "generation_batch_size": int(generation_batch_size),
        "n_layers": int(n_layers),
        "evaluation_batch_size": int(evaluation_batch_size),
        "formal_checkpoint_touched": False,
        "minimum_in_range_samples": int(minimum_in_range_samples),
        "minimum_in_range_per_pilot": int(minimum_in_range_per_pilot),
        "reference_initialization_fingerprint": reference_initialization_fingerprint,
        "system_seed": system_seed,
        "validation_plan_fingerprint": validation_plan_fingerprint(
            seed=seed,
            validation_samples=validation_samples,
            snr_db_range=candidate.snr_db_range,
            pilot_counts=tuple(candidate.evaluation.pilot_counts),
            n_layers=n_layers,
            evaluation_batch_size=evaluation_batch_size,
            channel_model=candidate.evaluation.channel_model,
            speed_mps_range=tuple(candidate.evaluation.speed_mps_range),
        ),
    }
    try:
        random.seed(seed)
        torch.manual_seed(seed)
        model = EqDeepRx(candidate)
        record["copied_initialization_tensors"] = _apply_reference_state(model, reference_state)
        _reset_system_rng(seed)
        if backend == "sionna":
            from eqdeeprx.sionna_system import SionnaTR38901System

            system = SionnaTR38901System(candidate, device=device)
        else:
            from eqdeeprx.signal import OFDMSystem

            system = OFDMSystem(candidate, device=device)
        history = train_steps(
            model,
            system,
            candidate,
            steps=training_steps,
            batch_size=batch_size,
            microbatch_size=microbatch_size,
            generation_batch_size=generation_batch_size,
            n_layers=n_layers,
            seed=seed,
            device=device,
            output_path=None,
        )
        _reset_system_rng(seed)
        metrics = _evaluate(
            model,
            system,
            candidate,
            backend=backend,
            seed=seed,
            n_layers=n_layers,
            validation_samples=validation_samples,
            evaluation_batch_size=evaluation_batch_size,
        )
        metric = _mean_eqdeeprx_ber(metrics)
        coverage_ok, coverage_reason = assess_validation_coverage(
            metrics,
            minimum_in_range_samples=minimum_in_range_samples,
            minimum_in_range_per_pilot=minimum_in_range_per_pilot,
        )
        history_finite = bool(history["losses"] and history["bers"])
        history_finite = history_finite and all(
            math.isfinite(float(item)) for item in (history["losses"][-1], history["bers"][-1])
        )
        status = "complete" if coverage_ok and metric is not None and history_finite else "nonfinite_metric"
        record.update(
            {
                "status": status,
                "history_steps": int(history["steps"]),
                "latest_loss": float(history["losses"][-1]),
                "latest_ber": float(history["bers"][-1]),
                "validation_mean_eqdeeprx_ber": metric,
                "validation_metrics": metrics,
                "validation_coverage_ok": coverage_ok,
                "validation_coverage_reason": coverage_reason,
                "finite": status == "complete",
            }
        )
    except Exception as exc:
        record.update(
            {
                "status": "error",
                "finite": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    finally:
        if "model" in locals():
            del model
        if "system" in locals():
            del system
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return record


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_matching_record(
    path: Path,
    *,
    profile_name: str,
    profile: dict[str, Any],
    seed: int,
    steps: int,
    backend: str | None = None,
    expected_config_fingerprint: str | None = None,
    validation_samples: int | None = None,
    batch_size: int | None = None,
    microbatch_size: int | None = None,
    generation_batch_size: int | None = None,
    n_layers: int | None = None,
    evaluation_batch_size: int | None = None,
    minimum_in_range_samples: int | None = None,
    minimum_in_range_per_pilot: int | None = None,
    expected_system_seed: int | None = None,
    expected_validation_plan_fingerprint: str | None = None,
) -> dict[str, Any] | None:
    """Load a resumable record only when its complete provenance matches.

    A record from an earlier profile or protocol must never be mistaken for a
    completed joint job merely because its filename and seed happen to match.
    The exact profile and its fingerprint are checked in addition to the
    finite/coverage checks used by the summary builder.
    """

    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict):
        return None
    try:
        record_seed = int(record.get("seed", -1))
        record_steps = int(record.get("training_steps", -1))
    except (TypeError, ValueError):
        return None
    if (
        record.get("joint_schema_version") != JOINT_SCHEMA_VERSION
        or record.get("profile_name") != profile_name
        or record.get("profile") != profile
        or record.get("profile_fingerprint") != profile_fingerprint(profile)
        or record_seed != int(seed)
        or record_steps != int(steps)
        or not _finite_record(record, steps)
    ):
        return None
    expected_fields = {
        "backend": backend,
        "validation_samples": validation_samples,
        "batch_size": batch_size,
        "microbatch_size": microbatch_size,
        "generation_batch_size": generation_batch_size,
        "n_layers": n_layers,
        "evaluation_batch_size": evaluation_batch_size,
        "minimum_in_range_samples": minimum_in_range_samples,
        "minimum_in_range_per_pilot": minimum_in_range_per_pilot,
        "system_seed": expected_system_seed,
        "validation_plan_fingerprint": expected_validation_plan_fingerprint,
        "config_fingerprint": expected_config_fingerprint,
    }
    for field, expected in expected_fields.items():
        if expected is None:
            continue
        actual = record.get(field)
        if field in {
            "validation_samples",
            "batch_size",
            "microbatch_size",
            "generation_batch_size",
            "n_layers",
            "evaluation_batch_size",
            "minimum_in_range_samples",
            "minimum_in_range_per_pilot",
            "system_seed",
        }:
            try:
                if int(actual) != int(expected):
                    return None
            except (TypeError, ValueError):
                return None
        elif actual != expected:
            return None
    return record


def _candidate_for_seed(
    config: EqDeepRxConfig, profile: dict[str, Any], seed: int
) -> EqDeepRxConfig:
    """Build the exact configuration used by one joint profile/seed job."""

    profile_config = apply_profile(config, profile)
    return replace(
        profile_config,
        training=replace(profile_config.training, seed=int(seed)),
    )


def _load_matching(path: Path, profile_name: str, seed: int, steps: int) -> dict[str, Any] | None:
    """Backward-compatible wrapper for callers using a registered profile."""

    profile = JOINT_PROFILES.get(profile_name)
    if profile is None:
        return None
    return load_matching_record(
        path,
        profile_name=profile_name,
        profile=profile,
        seed=seed,
        steps=steps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", default=",".join(JOINT_PROFILES))
    parser.add_argument("--steps", type=int, default=MIN_CONFIRMATION_STEPS)
    parser.add_argument("--seeds", default="2026,2027,2028")
    parser.add_argument("--backend", choices=("sionna", "fast"), default="sionna")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--microbatch-size", type=int, default=4)
    parser.add_argument("--generation-batch-size", type=int, default=2)
    parser.add_argument("--validation-samples", type=int, default=400)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--evaluation-batch-size", type=int, default=2)
    parser.add_argument("--min-in-range-samples", type=int, default=100)
    parser.add_argument("--min-in-range-per-pilot", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="stop cleanly after this many complete profile/seed records",
    )
    args = parser.parse_args()
    profile_names = [item.strip() for item in args.profiles.split(",") if item.strip()]
    unknown = [name for name in profile_names if name not in JOINT_PROFILES]
    if unknown:
        raise SystemExit(f"unknown profiles: {unknown}")
    seeds = tuple(int(item) for item in args.seeds.split(",") if item.strip())
    if args.steps < MIN_CONFIRMATION_STEPS:
        raise SystemExit(f"joint confirmation requires at least {MIN_CONFIRMATION_STEPS} steps")
    if args.max_records is not None and args.max_records < 1:
        raise SystemExit("--max-records must be positive when supplied")
    config = paper_config()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for seed in seeds:
        reference_config = replace(config, training=replace(config.training, seed=seed))
        reference_state, reference_fingerprint = _reference_model_state(reference_config, seed)
        for name in profile_names:
            path = args.output_dir / f"{name}_seed{seed}.json"
            candidate_config = _candidate_for_seed(config, JOINT_PROFILES[name], seed)
            record = (
                load_matching_record(
                    path,
                    profile_name=name,
                    profile=JOINT_PROFILES[name],
                    seed=seed,
                    steps=args.steps,
                    backend=args.backend,
                    expected_config_fingerprint=config_fingerprint(candidate_config),
                    validation_samples=args.validation_samples,
                    batch_size=args.batch_size,
                    microbatch_size=args.microbatch_size,
                    generation_batch_size=args.generation_batch_size,
                    n_layers=args.n_layers,
                    evaluation_batch_size=args.evaluation_batch_size,
                    minimum_in_range_samples=args.min_in_range_samples,
                    minimum_in_range_per_pilot=args.min_in_range_per_pilot,
                    expected_system_seed=_candidate_system_seed(seed),
                    expected_validation_plan_fingerprint=validation_plan_fingerprint(
                        seed=seed,
                        validation_samples=args.validation_samples,
                        snr_db_range=candidate_config.snr_db_range,
                        pilot_counts=tuple(candidate_config.evaluation.pilot_counts),
                        n_layers=args.n_layers,
                        evaluation_batch_size=args.evaluation_batch_size,
                        channel_model=candidate_config.evaluation.channel_model,
                        speed_mps_range=tuple(candidate_config.evaluation.speed_mps_range),
                    ),
                )
                if args.resume
                else None
            )
            if record is None:
                record = run_profile(
                    config,
                    profile_name=name,
                    profile=JOINT_PROFILES[name],
                    seed=seed,
                    backend=args.backend,
                    device=args.device,
                    training_steps=args.steps,
                    batch_size=args.batch_size,
                    microbatch_size=args.microbatch_size,
                    generation_batch_size=args.generation_batch_size,
                    validation_samples=args.validation_samples,
                    n_layers=args.n_layers,
                    evaluation_batch_size=args.evaluation_batch_size,
                    minimum_in_range_samples=args.min_in_range_samples,
                    minimum_in_range_per_pilot=args.min_in_range_per_pilot,
                    reference_state=reference_state,
                    reference_initialization_fingerprint=reference_fingerprint,
                )
                _write_json(path, record)
            records.append(record)
            summary = build_joint_summary(
                profiles=profile_names,
                records=records,
                seeds=seeds,
                minimum_steps=args.steps,
                profile_overrides={name: JOINT_PROFILES[name] for name in profile_names},
            )
            _write_json(args.output_dir / "joint_summary.json", summary)
            completed_record_count = sum(
                record.get("status") == "complete" for record in records
            )
            if reached_record_limit(completed_record_count, args.max_records):
                print(json.dumps(summary, indent=2, allow_nan=False))
                return
    summary = build_joint_summary(
        profiles=profile_names,
        records=records,
        seeds=seeds,
        minimum_steps=args.steps,
        profile_overrides={name: JOINT_PROFILES[name] for name in profile_names},
    )
    _write_json(args.output_dir / "joint_summary.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
