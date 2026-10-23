"""Evaluate the original 70k checkpoint with the corrected paper baseline.

Only eqdeeprx/evaluation.py may differ from the sources saved at training time.
The model, signal generator, receiver, configuration, and training code remain
byte-identical. The ordinary training-resume source guard stays strict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from eqdeeprx.__main__ import _new_model, _prepare_cuda
from eqdeeprx.config import paper_contract_errors, frozen_choice_errors
from eqdeeprx.eqdeeprx_runtime import configure_runtime_storage
from eqdeeprx.evaluation import _model_fingerprint, evaluate_paper_figure6a
from eqdeeprx.provenance import require_source_unchanged, source_fingerprint
from eqdeeprx.sionna_system import SionnaTR38901System
from eqdeeprx.training import formal_checkpoint_errors


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_training_sources(package: Path, checkpoint: Path, manifest: Path) -> dict:
    record = json.loads(manifest.read_text(encoding="utf-8-sig"))
    if sha256(checkpoint) != record["checkpoint_sha256"]:
        raise ValueError("checkpoint bytes differ from the completed 70k run")
    expected = record["package_file_sha256"]
    actual_names = {path.name for path in package.glob("*.py")}
    if actual_names != set(expected):
        raise ValueError("package module set differs from the 70k training sources")
    for name, original_hash in expected.items():
        if name != "evaluation.py" and sha256(package / name) != original_hash:
            raise ValueError(f"training source changed: {name}")
    if sha256(package / "evaluation.py") == expected["evaluation.py"]:
        raise ValueError("the corrected evaluation implementation was not loaded")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("D:/EqDeepRxRuns/strict_fig6a"))
    parser.add_argument("--samples", type=int, default=32_000)
    parser.add_argument("--layers", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.samples != 32_000:
        raise ValueError("formal Figure 6(a) uses 32000 independent slots per pilot count")
    package = Path(__file__).resolve().parent / "eqdeeprx"
    checkpoint_path = args.run_dir / "eqdeeprx.pt"
    manifest_path = args.run_dir / "training_source_manifest.json"
    record = verify_training_sources(package, checkpoint_path, manifest_path)
    configure_runtime_storage(args.run_dir / "runtime")
    device = _prepare_cuda()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("source_fingerprint") != record["source_fingerprint"]:
        raise ValueError("checkpoint training source does not match the saved manifest")
    config = checkpoint["config"]
    if paper_contract_errors(config) or frozen_choice_errors(config):
        raise ValueError("checkpoint configuration does not satisfy the paper contract")
    model = _new_model(config, device)
    signature = checkpoint["run_signature"]
    errors = formal_checkpoint_errors(
        checkpoint,
        model,
        config,
        expected_steps=config.training.total_steps,
        batch_size=config.training.batch_size,
        microbatch_size=signature["microbatch_size"],
        generation_batch_size=signature["generation_batch_size"],
        seed=config.training.seed,
        expected_amp=config.training.amp_dtype == "bfloat16",
        expected_device=device,
    )
    expected_mismatch = "checkpoint source does not match this audited implementation"
    if errors != [expected_mismatch]:
        raise ValueError("invalid checkpoint: " + "; ".join(errors or ["source mismatch was not detected"]))
    if checkpoint["next_step"] != 70_000:
        raise ValueError("checkpoint has not completed 70000 steps")
    model.load_state_dict(checkpoint["model_state_dict"])
    model_hash = _model_fingerprint(model)
    evaluation_hash = require_source_unchanged()
    print(json.dumps({
        "checkpoint_next_step": checkpoint["next_step"],
        "checkpoint_sha256": record["checkpoint_sha256"],
        "checkpoint_training_source_fingerprint": record["source_fingerprint"],
        "evaluation_source_fingerprint": evaluation_hash,
        "model_fingerprint": model_hash,
        "n_layers": args.layers,
        "samples_per_pilot": args.samples,
        "verify_only": args.verify_only,
    }), flush=True)
    if args.verify_only:
        return
    metrics = evaluate_paper_figure6a(
        model,
        SionnaTR38901System(config, device),
        config,
        validation_samples=args.samples,
        evaluation_batch_size=2,
        n_layers=args.layers,
        seed=config.training.seed + 1_000_000,
        resume=args.resume,
        diagnostic=False,
        output_dir=args.run_dir / f"figure6a_{args.layers}layers_corrected",
        checkpoint_provenance={
            "checkpoint_next_step": checkpoint["next_step"],
            "checkpoint_status": checkpoint["checkpoint_status"],
            "checkpoint_model_fingerprint": model_hash,
            "checkpoint_training_source_fingerprint": record["source_fingerprint"],
            "checkpoint_sha256": record["checkpoint_sha256"],
        },
    )
    print(json.dumps({"evaluation_complete": True,
                      "validation_samples_total": metrics["validation_samples_total"],
                      "output_dir": str(args.run_dir / f"figure6a_{args.layers}layers_corrected")}), flush=True)


if __name__ == "__main__":
    main()
