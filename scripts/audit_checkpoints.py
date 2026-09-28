from __future__ import annotations

"""Read-only checkpoint contract auditor for formal EqDeepRx runs."""

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from eqdeeprx.config import EqDeepRxConfig, paper_config
from eqdeeprx.model import EqDeepRx
from eqdeeprx.training import (
    Lamb,
    _checkpoint_consistency_errors,
    _finite_tensor_tree,
    _scalar_step,
    config_fingerprint,
)


def _optimizer_step_counts(payload: dict[str, Any]) -> dict[str, int]:
    state = payload.get("optimizer_state_dict", {}).get("state", {})
    counts = Counter()
    for entry in state.values() if isinstance(state, dict) else ():
        step = _scalar_step(entry.get("step")) if isinstance(entry, dict) else None
        counts[str(step)] += 1
    return dict(sorted(counts.items()))


def audit_payload(
    payload: dict[str, Any],
    *,
    config: EqDeepRxConfig,
    run_signature: dict[str, Any],
    amp_enabled: bool,
) -> dict[str, Any]:
    """Audit one already-loaded payload without changing it."""

    model = EqDeepRx(config)
    optimizer = Lamb(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=(config.training.lamb_beta1, config.training.lamb_beta2),
        eps=config.training.lamb_eps,
        weight_decay=config.training.weight_decay,
        bias_correction=config.training.lamb_bias_correction,
    )
    reasons = _checkpoint_consistency_errors(
        payload,
        model,
        optimizer,
        expected_config=config,
        expected_signature=run_signature,
        expected_amp=amp_enabled,
    )
    history = payload.get("history")
    history_lengths = {
        name: len(history.get(name, [])) if isinstance(history, dict) and isinstance(history.get(name, []), (list, tuple)) else None
        for name in ("losses", "bers", "learning_rates")
    }
    model_state = payload.get("model_state_dict")
    optimizer_state = payload.get("optimizer_state_dict")
    scaler_state = payload.get("grad_scaler_state_dict")
    next_step = _scalar_step(payload.get("next_step"))
    report = {
        "file": None,
        "valid": not reasons,
        "reasons": reasons,
        "next_step": next_step,
        "steps": _scalar_step(payload.get("steps")),
        "history_steps": _scalar_step(history.get("steps")) if isinstance(history, dict) else None,
        "history_lengths": history_lengths,
        "optimizer_step_counts": _optimizer_step_counts(payload),
        "scaler_scale": scaler_state.get("scale") if isinstance(scaler_state, dict) else None,
        "model_finite": not bool(_finite_tensor_tree(model_state, "model_state_dict")) if isinstance(model_state, dict) else False,
        "optimizer_finite": not bool(_finite_tensor_tree(optimizer_state, "optimizer_state_dict")) if isinstance(optimizer_state, dict) else False,
        "checkpoint_status": payload.get("checkpoint_status"),
        "checkpoint_schema_version": payload.get("checkpoint_schema_version"),
        "config_fingerprint": config_fingerprint(payload.get("config")) if payload.get("config") is not None else None,
        "expected_config_fingerprint": config_fingerprint(config),
        "run_signature": payload.get("run_signature"),
    }
    return report


def audit_directory(
    root: Path,
    *,
    config: EqDeepRxConfig,
    run_signature: dict[str, Any],
    amp_enabled: bool,
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    for path in sorted(Path(root).rglob("*.pt")):
        if path.name.endswith(".tmp"):
            continue
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            report = audit_payload(
                payload,
                config=config,
                run_signature=run_signature,
                amp_enabled=amp_enabled,
            )
        except Exception as exc:  # malformed files are candidates with a reason
            report = {
                "file": None,
                "valid": False,
                "reasons": [f"load error: {type(exc).__name__}: {exc}"],
                "next_step": None,
            }
        report["file"] = str(path)
        reports.append(report)
    valid = sorted(
        (item for item in reports if item.get("valid")),
        key=lambda item: (item.get("next_step") is not None, item.get("next_step") or -1),
        reverse=True,
    )
    for rank, item in enumerate(valid, start=1):
        item["candidate_rank"] = rank
    for item in reports:
        item.setdefault("candidate_rank", None)
    return {
        "root": str(root),
        "config_fingerprint": config_fingerprint(config),
        "run_signature": run_signature,
        "amp_enabled": amp_enabled,
        "candidates": reports,
        "highest_valid": valid[0]["file"] if valid else None,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=70_000)
    parser.add_argument("--batch-size", type=int, default=112)
    parser.add_argument("--microbatch-size", type=int, default=28)
    parser.add_argument("--generation-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-amp", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = paper_config()
    signature = {
        "batch_size": args.batch_size,
        "generation_batch_size": args.generation_batch_size,
        "microbatch_size": args.microbatch_size,
        "n_layers": None,
        "pilot_count": None,
        "snr_db": None,
        "seed": args.seed,
    }
    result = audit_directory(
        args.root,
        config=config,
        run_signature=signature,
        amp_enabled=not args.no_amp,
    )
    result["requested_steps"] = args.steps
    result["seed"] = args.seed
    encoded = json.dumps(result, indent=2, default=str)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
