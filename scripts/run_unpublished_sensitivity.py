from __future__ import annotations

"""Run fixed-sample, one-variable exploratory sensitivity experiments.

This script never reads or writes a formal training checkpoint. Results are
exploratory and cannot be relabeled as the paper's strict reproduction.
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
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from eqdeeprx.config import EqDeepRxConfig, paper_config
from eqdeeprx.evaluation import (
    _model_fingerprint,
    evaluate_paper_figure6a,
    evaluate_uncoded_ber,
)
from eqdeeprx.model import EqDeepRx
from eqdeeprx.signal import OFDMSystem
from eqdeeprx.training import config_fingerprint, train_steps


DEFAULT_CANDIDATES: dict[str, tuple[Any, ...]] = {
    "time_mixer_channels": (1, 2, 4),
    # CDL validation uses a separate RMS delay-spread choice; the UMa
    # training range must not be silently reused for this protocol.
    "cdl_delay_spread_mode": (
        "uniform_10_1100ns",
        "fixed_300ns",
        "fixed_100ns",
        "fixed_1000ns",
    ),
    "uma_delay_spread_mode": ("native", "uniform_10_1100ns"),
    "cir_discretization": ("sinc", "nearest"),
    "cir_normalization": (True, False),
    "pathloss": (False, True),
    "shadow_fading": (False, True),
    "interferer_timing": ("random_symbol", "zero", "random_sample"),
    "lamb_beta1": (0.9, 0.8),
    "lamb_beta2": (0.999, 0.99),
    "lamb_eps": (1e-6, 1e-8),
    "weight_decay": (0.0, 1e-4),
    "lamb_bias_correction": (True, False),
    "amp_dtype": ("bfloat16", "float32"),
    "vcl_attachment": ("all", "final", "none"),
    "symbol_loss_reduction": ("sum", "mean_active"),
    "residual_projection_bias": (False, True),
}

# A short exploratory run is useful only when it produces enough realized-SINR
# samples to compare candidates. Empty/out-of-range bins are a coverage
# failure, never a numerical score of zero or NaN.
DEFAULT_MIN_IN_RANGE_SAMPLES = 100
DEFAULT_MIN_IN_RANGE_PER_PILOT = 20
SENSITIVITY_SCHEMA_VERSION = 5
MIN_CONFIRMATION_STEPS = 2_000
VALIDATION_PLAN_SCHEMA_VERSION = 1
SYSTEM_SEED_MULTIPLIER = 1_000_003
SYSTEM_SEED_OFFSET = 17_113
SYSTEM_SEED_MODULUS = 2_147_483_647


def _candidate_system_seed(seed: int) -> int:
    """Derive a channel-library seed independent of model initialization."""

    value = (int(seed) * SYSTEM_SEED_MULTIPLIER + SYSTEM_SEED_OFFSET) % SYSTEM_SEED_MODULUS
    return value if value > 0 else SYSTEM_SEED_OFFSET


def _reset_system_rng(seed: int) -> int:
    """Reset process/library RNGs before constructing or evaluating a system."""

    system_seed = _candidate_system_seed(seed)
    random.seed(system_seed)
    torch.manual_seed(system_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(system_seed)
    try:
        import numpy as np

        np.random.seed(system_seed)
    except ImportError:
        pass
    try:
        from sionna.phy import config as sionna_config

        sionna_config.seed = system_seed
    except ImportError:
        pass
    return system_seed


def validation_plan_fingerprint(
    *,
    seed: int,
    validation_samples: int,
    snr_db_range: tuple[float, float] = (0.0, 45.0),
    pilot_counts: tuple[int, ...] = (1, 2),
    n_layers: int = 4,
    evaluation_batch_size: int = 2,
    channel_model: str = "CDL-C",
    speed_mps_range: tuple[float, float] = (10.0, 15.0),
) -> str:
    """Hash the fixed sample IDs and the complete evaluation protocol.

    Candidate values are intentionally absent from this payload: every
    candidate must consume the same sample IDs while its one changed choice
    alters the generated channel.  Protocol fields are included so a record
    from a diagnostic layer count, batch size, or validation scenario cannot
    be reused as a formal comparison.
    """

    if validation_samples <= 0:
        raise ValueError("validation_samples must be positive")
    if not pilot_counts:
        raise ValueError("pilot_counts must not be empty")
    if n_layers < 1 or evaluation_batch_size < 1:
        raise ValueError("n_layers and evaluation_batch_size must be positive")
    if len(speed_mps_range) != 2 or speed_mps_range[0] < 0 or speed_mps_range[1] < speed_mps_range[0]:
        raise ValueError("speed_mps_range must be a nonnegative ordered pair")
    samples_per_pilot, remainder = divmod(validation_samples, len(pilot_counts))
    plan = []
    for index, pilot_count in enumerate(pilot_counts):
        count = samples_per_pilot + int(index < remainder)
        for sample in range(count):
            sample_seed = int(seed) + int(pilot_count) * 1_000_000 + sample
            requested_snr = random.Random(sample_seed).uniform(*snr_db_range)
            plan.append(
                {
                    "pilot_count": int(pilot_count),
                    "sample_index": int(sample),
                    "sample_seed": sample_seed,
                    "snr_db": float(requested_snr),
                }
            )
    payload = {
        "schema": VALIDATION_PLAN_SCHEMA_VERSION,
        "seed": int(seed),
        "validation_samples": int(validation_samples),
        "snr_db_range": [float(snr_db_range[0]), float(snr_db_range[1])],
        "pilot_counts": [int(value) for value in pilot_counts],
        "n_layers": int(n_layers),
        "evaluation_batch_size": int(evaluation_batch_size),
        "channel_model": str(channel_model),
        "speed_mps_range": [float(speed_mps_range[0]), float(speed_mps_range[1])],
        "plan": plan,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _parse_scalar(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        number = float(value)
    except ValueError:
        return value.strip()
    return int(number) if number.is_integer() else number


def parse_candidate_values(text: str, variable: str) -> list[Any]:
    """Parse CLI values while preserving string and boolean choices."""

    values = [_parse_scalar(item) for item in text.split(",") if item.strip()]
    if variable in {"time_mixer_channels"}:
        values = [int(value) for value in values]
    elif variable == "cdl_delay_spread_mode":
        values = [str(value) for value in values]
    elif variable in {
        "cir_normalization",
        "pathloss",
        "shadow_fading",
        "lamb_bias_correction",
        "residual_projection_bias",
    }:
        values = [bool(value) for value in values]
    elif variable in {
        "lamb_beta1",
        "lamb_beta2",
        "lamb_eps",
        "weight_decay",
    }:
        values = [float(value) for value in values]
    return values


def apply_candidate(config: EqDeepRxConfig, variable: str, value: Any) -> EqDeepRxConfig:
    """Return a config with exactly one unpublished choice changed."""

    if variable == "time_mixer_channels":
        return replace(config, model=replace(config.model, time_mixer_channels=int(value)))
    if variable == "residual_projection_bias":
        return replace(
            config,
            model=replace(config.model, residual_projection_bias=bool(value)),
        )
    if variable == "cdl_delay_spread_mode":
        if value == "uniform_10_1100ns":
            channel = replace(config.channel, cdl_delay_spread_mode="uniform_10_1100ns")
        elif str(value).startswith("fixed_") and str(value).endswith("ns"):
            channel = replace(config.channel, cdl_delay_spread_mode="fixed")
            fixed_ns = float(str(value)[len("fixed_"):-len("ns")])
            return replace(
                config,
                cdl_delay_spread_ns=fixed_ns,
                channel=channel,
            )
        else:
            raise ValueError(f"unsupported CDL delay-spread candidate: {value}")
        return replace(config, channel=channel)
    if variable in {
        "uma_delay_spread_mode",
        "cir_discretization",
        "cir_normalization",
        "interferer_timing",
    }:
        field = {
            "uma_delay_spread_mode": "uma_delay_spread_mode",
            "cir_discretization": "cir_discretization",
            "cir_normalization": "cir_normalization",
            "interferer_timing": "interferer_timing",
        }[variable]
        return replace(config, channel=replace(config.channel, **{field: value}))
    if variable == "pathloss":
        return replace(config, channel=replace(config.channel, enable_pathloss=bool(value)))
    if variable == "shadow_fading":
        return replace(config, channel=replace(config.channel, enable_shadow_fading=bool(value)))
    if variable in {"lamb_beta1", "lamb_beta2", "lamb_eps", "weight_decay"}:
        return replace(config, training=replace(config.training, **{variable: float(value)}))
    if variable == "lamb_bias_correction":
        return replace(
            config,
            training=replace(config.training, lamb_bias_correction=bool(value)),
        )
    if variable in {"amp_dtype", "vcl_attachment", "symbol_loss_reduction"}:
        return replace(config, training=replace(config.training, **{variable: value}))
    raise ValueError(f"unknown sensitivity variable: {variable}")


def formal_baseline_value(
    config: EqDeepRxConfig, variable: str, values: list[Any] | tuple[Any, ...]
) -> Any:
    """Return the candidate value that exactly preserves the formal config."""

    expected = config_fingerprint(config)
    matches = [
        value
        for value in values
        if config_fingerprint(apply_candidate(config, variable, value)) == expected
    ]
    if len(matches) != 1:
        raise ValueError(
            f"candidate matrix for {variable!r} must contain exactly one formal baseline; "
            f"matched {matches!r}"
        )
    return matches[0]


def _mean_eqdeeprx_ber(metrics: dict[str, Any]) -> float | None:
    """Return a pooled BER, or ``None`` when no finite evidence exists."""

    errors = metrics.get("curve_error_counts", {})
    bits = metrics.get("curve_bit_counts", {})
    total_errors = 0.0
    total_bits = 0.0
    for name in metrics.get("curves", {}):
        if not name.startswith("eqdeeprx_"):
            continue
        if name in errors and name in bits:
            for error, bit_count in zip(errors[name], bits[name]):
                if math.isfinite(float(error)) and math.isfinite(float(bit_count)):
                    total_errors += float(error)
                    total_bits += float(bit_count)
        else:
            # Backward-compatible fallback for metrics produced before the
            # pooled counters were added. Weight by populated sample counts.
            suffix = "1_pilot" if name.endswith("1_pilot") else "2_pilots"
            counts = metrics.get("sinr_bin_sample_counts", {}).get(suffix, [])
            for value, count in zip(metrics["curves"].get(name, []), counts):
                if value is not None and math.isfinite(float(value)) and count:
                    total_errors += float(value) * int(count)
                    total_bits += float(count)
    if total_bits <= 0.0:
        return None
    result = total_errors / total_bits
    return float(result) if math.isfinite(result) else None


def assess_validation_coverage(
    metrics: dict[str, Any],
    *,
    minimum_in_range_samples: int,
    minimum_in_range_per_pilot: int,
) -> tuple[bool, str | None]:
    """Check that a validation result has enough displayed-SINR evidence."""

    total = int(metrics.get("in_range_sample_count", 0))
    by_pilot = metrics.get("in_range_sample_count_by_pilot", {})
    if total < minimum_in_range_samples:
        return False, (
            f"in-range samples {total} < required {minimum_in_range_samples}"
        )
    missing = [
        str(pilot)
        for pilot, count in by_pilot.items()
        if int(count) < minimum_in_range_per_pilot
    ]
    if missing:
        return False, "pilot coverage below threshold for " + ", ".join(missing)
    return True, None


def summarize_candidate_decisions(
    records: list[dict[str, Any]],
    *,
    screening_seeds: tuple[int, ...] | list[int],
    confirmation_seed: int,
    minimum_confirmation_steps: int = MIN_CONFIRMATION_STEPS,
) -> list[dict[str, Any]]:
    """Summarize the preregistered multi-seed acceptance rule.

    A candidate is only a confirmed improvement when it has finite, complete
    records and beats the same-variable baseline for every screening seed and
    the confirmation seed. Missing, errored, or non-improving records are
    retained as explicit rejection reasons instead of being silently ignored.
    """

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (str(record.get("variable")), repr(record.get("value")))
        grouped.setdefault(key, []).append(record)

    screening = tuple(int(seed) for seed in screening_seeds)
    decisions: list[dict[str, Any]] = []
    for (variable, value_repr), items in grouped.items():
        value = items[0].get("value") if items else None
        baseline_values = {
            repr(item.get("baseline_value"))
            for item in items
            if "baseline_value" in item
        }
        is_baseline = bool(baseline_values) and value_repr in baseline_values
        by_seed = {int(item["seed"]): item for item in items if "seed" in item}
        reasons: list[str] = []
        screening_improvements: list[int] = []

        def inspect_seed(seed: int, phase: str) -> dict[str, Any] | None:
            record = by_seed.get(seed)
            if record is None:
                reasons.append(f"missing {phase} seed {seed}")
                return None
            if record.get("status") != "complete" or not record.get("finite", False):
                reasons.append(f"nonfinite or incomplete seed {seed}")
                return record
            if phase == "confirmation":
                recorded_steps = record.get(
                    "training_steps", record.get("screening_steps")
                )
                try:
                    enough_steps = int(recorded_steps) >= int(minimum_confirmation_steps)
                except (TypeError, ValueError):
                    enough_steps = False
                if not enough_steps:
                    reasons.append(
                        "confirmation steps below "
                        f"{int(minimum_confirmation_steps)}"
                    )
                    return record
            if not record.get("improves_baseline", False):
                reasons.append(f"seed {seed} did not improve baseline")
            return record

        for seed in screening:
            record = inspect_seed(seed, "screening")
            if (
                record is not None
                and record.get("status") == "complete"
                and record.get("finite", False)
                and record.get("improves_baseline", False)
            ):
                screening_improvements.append(seed)
        confirmation = inspect_seed(confirmation_seed, "confirmation")
        confirmation_steps_ok = False
        if confirmation is not None:
            try:
                confirmation_steps_ok = int(
                    confirmation.get(
                        "training_steps", confirmation.get("screening_steps")
                    )
                ) >= int(minimum_confirmation_steps)
            except (TypeError, ValueError):
                confirmation_steps_ok = False
        confirmation_improvement = bool(
            confirmation is not None
            and confirmation.get("status") == "complete"
            and confirmation.get("finite", False)
            and confirmation_steps_ok
            and confirmation.get("improves_baseline", False)
        )

        if is_baseline:
            decision = "baseline"
            reasons = []
        elif len(screening_improvements) == len(screening) and confirmation_improvement:
            decision = "confirmed_improvement"
            reasons = []
        elif len(screening_improvements) == len(screening):
            decision = "screening_improvement_only"
        else:
            decision = "rejected"

        decisions.append(
            {
                "variable": variable,
                "value": value,
                "decision": decision,
                "screening_improvements": screening_improvements,
                "confirmation_improvement": confirmation_improvement,
                "rejection_reasons": reasons,
                "seeds_present": sorted(by_seed),
                "metrics_by_seed": {
                    str(seed): by_seed[seed].get("validation_mean_eqdeeprx_ber")
                    for seed in sorted(by_seed)
                },
            }
        )
    return sorted(decisions, key=lambda item: (item["variable"], repr(item["value"])))


def build_sensitivity_summary(
    *,
    variables: tuple[str, ...] | list[str],
    matrix: dict[str, list[Any]],
    screening_seeds: tuple[int, ...] | list[int],
    confirmation_seed: int,
    coverage_policy: dict[str, int],
    records: list[dict[str, Any]],
    minimum_confirmation_steps: int = MIN_CONFIRMATION_STEPS,
) -> dict[str, Any]:
    """Build the persisted summary, including explicit candidate decisions."""

    expected = [
        (variable, value, int(seed))
        for variable in variables
        for value in matrix[variable]
        for seed in (*[int(item) for item in screening_seeds], int(confirmation_seed))
    ]
    present = {
        (str(record.get("variable")), record.get("value"), int(record.get("seed")))
        for record in records
        if record.get("variable") is not None and record.get("seed") is not None
    }
    missing = [
        {"variable": variable, "value": value, "seed": seed}
        for variable, value, seed in expected
        if (variable, value, seed) not in present
    ]
    return {
        "sensitivity_schema_version": SENSITIVITY_SCHEMA_VERSION,
        "status": "complete" if not missing else "incomplete",
        "execution_complete": not missing,
        "expected_record_count": len(expected),
        "missing_records": missing,
        "variables": list(variables),
        "candidate_matrix": matrix,
        "screening_seeds": [int(seed) for seed in screening_seeds],
        "confirmation_seed": int(confirmation_seed),
        "minimum_confirmation_steps": int(minimum_confirmation_steps),
        "coverage_policy": coverage_policy,
        "accepted_record_count": sum(
            bool(record.get("finite")) for record in records
        ),
        "records": records,
        "candidate_decisions": summarize_candidate_decisions(
            records,
            screening_seeds=screening_seeds,
            confirmation_seed=confirmation_seed,
            minimum_confirmation_steps=minimum_confirmation_steps,
        ),
        "formal_checkpoint_touched": False,
    }


def _reference_model_state(
    config: EqDeepRxConfig, seed: int
) -> tuple[dict[str, torch.Tensor], str]:
    """Build one deterministic initialization to share across candidates."""

    random.seed(seed)
    torch.manual_seed(seed)
    model = EqDeepRx(config)
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    return state, _model_fingerprint(model)


def _apply_reference_state(
    model: EqDeepRx,
    reference_state: dict[str, torch.Tensor] | None,
) -> int:
    """Copy unchanged named tensors while retaining candidate-specific shapes."""

    if reference_state is None:
        return 0
    current = model.state_dict()
    copied = 0
    for name, value in reference_state.items():
        if name in current and tuple(current[name].shape) == tuple(value.shape):
            current[name].copy_(value.to(dtype=current[name].dtype))
            copied += 1
    model.load_state_dict(current, strict=True)
    return copied


def _evaluate(
    model: EqDeepRx,
    system: Any,
    config: EqDeepRxConfig,
    *,
    backend: str,
    seed: int,
    n_layers: int,
    validation_samples: int,
    evaluation_batch_size: int,
) -> dict[str, Any]:
    if backend == "sionna":
        return evaluate_paper_figure6a(
            model,
            system,
            config,
            validation_samples=validation_samples,
            evaluation_batch_size=evaluation_batch_size,
            n_layers=n_layers,
            seed=seed,
            output_dir=None,
        )
    return evaluate_uncoded_ber(
        model,
        system,
        config,
        snr_points=(0.0, 6.0, 12.0),
        samples_per_point=max(1, validation_samples // 3),
        n_layers=n_layers,
        seed=seed,
        output_dir=None,
    )


def run_one(
    config: EqDeepRxConfig,
    *,
    variable: str,
    value: Any,
    seed: int,
    backend: str,
    device: str,
    screening_steps: int,
    training_steps: int | None = None,
    batch_size: int,
    microbatch_size: int,
    generation_batch_size: int,
    validation_samples: int,
    n_layers: int,
    evaluation_batch_size: int = 2,
    minimum_in_range_samples: int,
    minimum_in_range_per_pilot: int,
    reference_state: dict[str, torch.Tensor] | None = None,
    reference_initialization_fingerprint: str | None = None,
) -> dict[str, Any]:
    candidate = apply_candidate(config, variable, value)
    candidate = replace(candidate, training=replace(candidate.training, seed=seed))
    actual_steps = int(screening_steps if training_steps is None else training_steps)
    system_seed = _candidate_system_seed(seed)
    record: dict[str, Any] = {
        "sensitivity_schema_version": SENSITIVITY_SCHEMA_VERSION,
        "variable": variable,
        "value": value,
        "seed": seed,
        "config_fingerprint": config_fingerprint(candidate),
        "backend": backend,
        "screening_steps": int(screening_steps),
        # Keep the actual optimizer-step count explicit so confirmation runs
        # cannot be mistaken for a short screening record when directories are
        # reused or inspected together.
        "training_steps": actual_steps,
        "validation_samples": validation_samples,
        "batch_size": batch_size,
        "microbatch_size": microbatch_size,
        "generation_batch_size": generation_batch_size,
        "n_layers": n_layers,
        "evaluation_batch_size": evaluation_batch_size,
        "formal_checkpoint_touched": False,
        "minimum_in_range_samples": minimum_in_range_samples,
        "minimum_in_range_per_pilot": minimum_in_range_per_pilot,
        "reference_initialization_fingerprint": reference_initialization_fingerprint,
        "system_seed": system_seed,
        "validation_plan_fingerprint": validation_plan_fingerprint(
            seed=seed,
            validation_samples=validation_samples,
            snr_db_range=config.snr_db_range,
            pilot_counts=tuple(config.evaluation.pilot_counts),
            n_layers=n_layers,
            evaluation_batch_size=evaluation_batch_size,
            channel_model=config.evaluation.channel_model,
            speed_mps_range=tuple(config.evaluation.speed_mps_range),
        ),
    }
    try:
        random.seed(seed)
        torch.manual_seed(seed)
        model = EqDeepRx(candidate)
        record["copied_initialization_tensors"] = _apply_reference_state(
            model, reference_state
        )
        # Construct the channel system from an RNG stream that is independent
        # of candidate-specific model shapes and initialization draws.
        _reset_system_rng(seed)
        if backend == "sionna":
            from eqdeeprx.sionna_system import SionnaTR38901System

            system = SionnaTR38901System(candidate, device=device)
        else:
            system = OFDMSystem(candidate, device=device)
        history = train_steps(
            model,
            system,
            candidate,
            steps=actual_steps,
            batch_size=batch_size,
            microbatch_size=microbatch_size,
            generation_batch_size=generation_batch_size,
            n_layers=n_layers,
            seed=seed,
            device=device,
            output_path=None,
        )
        # Rewind the library RNG before validation as well.  The evaluator
        # derives per-sample seeds, but Sionna also has process-level state.
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
        metrics["system_seed"] = system_seed
        metrics["validation_plan_fingerprint"] = record[
            "validation_plan_fingerprint"
        ]
        metric = _mean_eqdeeprx_ber(metrics)
        has_coverage, coverage_reason = assess_validation_coverage(
            metrics,
            minimum_in_range_samples=minimum_in_range_samples,
            minimum_in_range_per_pilot=minimum_in_range_per_pilot,
        )
        history_finite = all(
            math.isfinite(float(item))
            for item in (history["losses"][-1], history["bers"][-1])
        )
        if not has_coverage:
            status = "insufficient_validation_coverage"
        elif metric is None or not history_finite:
            status = "nonfinite_metric"
        else:
            status = "complete"
        record.update(
            {
                "status": status,
                "history_steps": history["steps"],
                "latest_loss": float(history["losses"][-1]),
                "latest_ber": float(history["bers"][-1]),
                "validation_mean_eqdeeprx_ber": metric,
                "validation_metrics": metrics,
                "validation_coverage_ok": has_coverage,
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
    return record


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_existing_record(
    path: Path,
    *,
    variable: str,
    value: Any,
    seed: int,
    screening_steps: int,
    validation_samples: int,
    expected_training_steps: int | None = None,
    expected_config_fingerprint: str | None = None,
    expected_system_seed: int | None = None,
    expected_validation_plan_fingerprint: str | None = None,
    expected_n_layers: int = 4,
    expected_evaluation_batch_size: int = 2,
    expected_channel_model: str = "CDL-C",
    expected_speed_mps_range: tuple[float, float] = (10.0, 15.0),
) -> dict[str, Any] | None:
    """Load one resumable result only when its execution contract matches.

    A JSON file can be left behind by an interrupted or older probe.  Reusing
    it is safe only when the candidate identity and run-size/configuration
    fields agree with the requested job.  Invalid or stale files return
    ``None`` so the caller reruns that candidate and rewrites only the
    exploratory result path.
    """

    path = Path(path)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("sensitivity_schema_version") != SENSITIVITY_SCHEMA_VERSION:
        return None
    if payload.get("variable") != variable:
        return None
    if payload.get("value") != value:
        return None
    try:
        if int(payload.get("seed")) != int(seed):
            return None
        if int(payload.get("screening_steps")) != int(screening_steps):
            return None
        if int(payload.get("validation_samples")) != int(validation_samples):
            return None
        if expected_training_steps is not None:
            recorded_steps = payload.get("training_steps", payload.get("screening_steps"))
            if int(recorded_steps) != int(expected_training_steps):
                return None
    except (TypeError, ValueError):
        return None
    if expected_config_fingerprint is not None and payload.get("config_fingerprint") != expected_config_fingerprint:
        return None
    try:
        recorded_system_seed = int(payload["system_seed"])
    except (KeyError, TypeError, ValueError):
        return None
    expected_system_seed = (
        _candidate_system_seed(seed)
        if expected_system_seed is None
        else int(expected_system_seed)
    )
    if recorded_system_seed != expected_system_seed:
        return None
    recorded_plan = payload.get("validation_plan_fingerprint")
    expected_plan = (
        validation_plan_fingerprint(
            seed=seed,
            validation_samples=validation_samples,
            n_layers=expected_n_layers,
            evaluation_batch_size=expected_evaluation_batch_size,
            channel_model=expected_channel_model,
            speed_mps_range=expected_speed_mps_range,
        )
        if expected_validation_plan_fingerprint is None
        else expected_validation_plan_fingerprint
    )
    if recorded_plan != expected_plan:
        return None
    if payload.get("status") not in {
        "complete",
        "insufficient_validation_coverage",
        "nonfinite_metric",
        "error",
    }:
        return None
    return payload


def _tiny_config() -> EqDeepRxConfig:
    base = paper_config().with_modulation("16QAM")
    return replace(
        base,
        n_subcarriers=16,
        n_fft=24,
        cyclic_prefix=4,
        n_rx_antennas=4,
        n_tx_antennas=4,
        layer_counts=(2, 3, 4),
        training=replace(base.training, layer_counts=(2, 3, 4)),
        model=replace(
            base.model,
            detector_channels=8,
            detector_sections=1,
            demapper_widths=(4, 4, 4, 8),
            denoise_widths=(8, 8, 8, 2),
            denoise_subsamples=(1, 2, 2, 1),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variable", choices=tuple(DEFAULT_CANDIDATES) + ("all",), required=True)
    parser.add_argument("--values", default="")
    parser.add_argument("--screening-steps", type=int, default=2000)
    parser.add_argument(
        "--confirmation-steps",
        type=int,
        default=None,
        help=(
            "optimizer steps for the confirmation seed; when omitted, the "
            "screening length is used and the result cannot pass the "
            f"{MIN_CONFIRMATION_STEPS}-step confirmation gate"
        ),
    )
    parser.add_argument("--seeds", default="2026,2027")
    parser.add_argument("--confirmation-seed", type=int, default=2028)
    parser.add_argument("--backend", choices=("fast", "sionna"), default="fast")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument(
        "--evaluation-batch-size",
        type=int,
        default=2,
        help="validation batch size; it is part of the resumable protocol fingerprint",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--microbatch-size", type=int, default=2)
    parser.add_argument("--generation-batch-size", type=int, default=2)
    parser.add_argument("--validation-samples", type=int, default=12)
    parser.add_argument(
        "--min-in-range-samples",
        type=int,
        default=DEFAULT_MIN_IN_RANGE_SAMPLES,
    )
    parser.add_argument(
        "--min-in-range-per-pilot",
        type=int,
        default=DEFAULT_MIN_IN_RANGE_PER_PILOT,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse matching completed candidate JSON files and continue missing jobs",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tiny", action="store_true", help="Use a small Sionna configuration for a structural smoke matrix")
    args = parser.parse_args()
    if min(
        args.screening_steps,
        args.validation_samples,
        args.min_in_range_samples,
        args.min_in_range_per_pilot,
        args.evaluation_batch_size,
    ) < 1:
        raise SystemExit(
            "steps, validation samples, and coverage thresholds must be positive"
        )
    if args.confirmation_steps is not None and args.confirmation_steps < 1:
        raise SystemExit("confirmation steps must be positive when supplied")
    variables = tuple(DEFAULT_CANDIDATES) if args.variable == "all" else (args.variable,)
    screening_seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    seeds = screening_seeds + [args.confirmation_seed]
    config = _tiny_config() if args.tiny else paper_config()
    matrix: dict[str, list[Any]] = {}
    for variable in variables:
        matrix[variable] = (
            parse_candidate_values(args.values, variable)
            if args.variable != "all" and args.values
            else list(DEFAULT_CANDIDATES[variable])
        )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "variables": variables,
                    "matrix": matrix,
                    "seeds": seeds,
                    "screening_steps": args.screening_steps,
                    "confirmation_steps": args.confirmation_steps,
                    "evaluation_batch_size": args.evaluation_batch_size,
                    "min_in_range_samples": args.min_in_range_samples,
                    "min_in_range_per_pilot": args.min_in_range_per_pilot,
                },
                indent=2,
            )
        )
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, Any]] = []

    def write_progress_summary() -> None:
        _write_json(
            args.output_dir / "sensitivity_summary.json",
            build_sensitivity_summary(
                variables=variables,
                matrix=matrix,
                screening_seeds=screening_seeds,
                confirmation_seed=args.confirmation_seed,
                minimum_confirmation_steps=MIN_CONFIRMATION_STEPS,
                coverage_policy={
                    "minimum_in_range_samples": args.min_in_range_samples,
                    "minimum_in_range_per_pilot": args.min_in_range_per_pilot,
                },
                records=all_records,
            ),
        )

    for variable in variables:
        values = matrix[variable]
        baseline = formal_baseline_value(config, variable, values)
        for seed in seeds:
            actual_steps = (
                args.confirmation_steps
                if seed == args.confirmation_seed and args.confirmation_steps is not None
                else args.screening_steps
            )
            baseline_config = replace(
                apply_candidate(config, variable, baseline),
                training=replace(config.training, seed=seed),
            )
            reference_state, reference_fingerprint = _reference_model_state(
                baseline_config, seed
            )
            baseline_path = args.output_dir / variable / f"{str(baseline).replace('/', '_')}_seed{seed}.json"
            baseline_record = None
            if args.resume:
                baseline_record = load_existing_record(
                    baseline_path,
                    variable=variable,
                    value=baseline,
                    seed=seed,
                    screening_steps=args.screening_steps,
                    validation_samples=args.validation_samples,
                    expected_training_steps=actual_steps,
                    expected_config_fingerprint=config_fingerprint(baseline_config),
                    expected_system_seed=_candidate_system_seed(seed),
                    expected_validation_plan_fingerprint=validation_plan_fingerprint(
                        seed=seed,
                        validation_samples=args.validation_samples,
                        snr_db_range=baseline_config.snr_db_range,
                        pilot_counts=tuple(baseline_config.evaluation.pilot_counts),
                        n_layers=args.n_layers,
                        evaluation_batch_size=args.evaluation_batch_size,
                        channel_model=baseline_config.evaluation.channel_model,
                        speed_mps_range=tuple(baseline_config.evaluation.speed_mps_range),
                    ),
                    expected_n_layers=args.n_layers,
                    expected_evaluation_batch_size=args.evaluation_batch_size,
                    expected_channel_model=baseline_config.evaluation.channel_model,
                    expected_speed_mps_range=tuple(baseline_config.evaluation.speed_mps_range),
                )
            if baseline_record is None:
                baseline_record = run_one(
                    config,
                    variable=variable,
                    value=baseline,
                    seed=seed,
                    backend=args.backend,
                    device=args.device,
                    screening_steps=args.screening_steps,
                    training_steps=actual_steps,
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
            baseline_metric = baseline_record.get("validation_mean_eqdeeprx_ber")
            for value in values:
                record_path = args.output_dir / variable / f"{str(value).replace('/', '_')}_seed{seed}.json"
                if value == baseline:
                    record = baseline_record
                else:
                    record = None
                    if args.resume:
                        candidate_config = replace(
                            apply_candidate(config, variable, value),
                            training=replace(config.training, seed=seed),
                        )
                        record = load_existing_record(
                            record_path,
                            variable=variable,
                            value=value,
                            seed=seed,
                            screening_steps=args.screening_steps,
                            validation_samples=args.validation_samples,
                            expected_training_steps=actual_steps,
                            expected_config_fingerprint=config_fingerprint(candidate_config),
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
                            expected_n_layers=args.n_layers,
                            expected_evaluation_batch_size=args.evaluation_batch_size,
                            expected_channel_model=candidate_config.evaluation.channel_model,
                            expected_speed_mps_range=tuple(candidate_config.evaluation.speed_mps_range),
                        )
                    if record is None:
                        record = run_one(
                            config,
                            variable=variable,
                            value=value,
                            seed=seed,
                            backend=args.backend,
                            device=args.device,
                            screening_steps=args.screening_steps,
                            training_steps=actual_steps,
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
                record = dict(record)
                record.setdefault("training_steps", int(actual_steps))
                record["phase"] = "confirmation" if seed == args.confirmation_seed else "screening"
                record["baseline_value"] = baseline
                record["baseline_validation_mean_eqdeeprx_ber"] = baseline_metric
                record["improves_baseline"] = bool(
                    record.get("status") == "complete"
                    and isinstance(baseline_metric, (int, float))
                    and math.isfinite(float(baseline_metric))
                    and isinstance(
                        record.get("validation_mean_eqdeeprx_ber"), (int, float)
                    )
                    and math.isfinite(
                        float(record["validation_mean_eqdeeprx_ber"])
                    )
                    and float(record["validation_mean_eqdeeprx_ber"])
                    < float(baseline_metric)
                )
                all_records.append(record)
                _write_json(
                    record_path,
                    record,
                )
                write_progress_summary()
    summary = build_sensitivity_summary(
        variables=variables,
        matrix=matrix,
        screening_seeds=screening_seeds,
        confirmation_seed=args.confirmation_seed,
        minimum_confirmation_steps=MIN_CONFIRMATION_STEPS,
        coverage_policy={
            "minimum_in_range_samples": args.min_in_range_samples,
            "minimum_in_range_per_pilot": args.min_in_range_per_pilot,
        },
        records=all_records,
    )
    _write_json(args.output_dir / "sensitivity_summary.json", summary)
    print(json.dumps(summary, indent=2, default=str, allow_nan=False))


if __name__ == "__main__":
    main()
