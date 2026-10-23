"""Windows/CUDA entry point for the audited Figure 6(a) study."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import statistics
from pathlib import Path

from .config import paper_config, paper_contract_errors, frozen_choice_errors
from .provenance import require_source_unchanged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EqDeepRx Figure 6(a): preflight, joint training, pre-decoding BER")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "train", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", type=Path, default=Path("D:/EqDeepRxRuns/strict_fig6a"))
        if name == "preflight":
            command.add_argument("--microbatch-size", type=int, default=56)
            command.add_argument("--generation-batch-size", type=int, default=2)
            command.add_argument("--repeats", type=int, default=3)
        elif name == "train":
            command.add_argument("--resume", action="store_true")
            command.add_argument("--confirm", action="store_true")
            command.add_argument("--steps", type=int, default=70_000)
            command.add_argument("--save-every", type=int, default=50)
        else:
            command.add_argument("--checkpoint", type=Path)
            command.add_argument("--resume", action="store_true")
            command.add_argument("--samples", type=int, default=32_000)
            command.add_argument("--layers", type=int, choices=(2, 3, 4), default=4,
                help="Use Table I's default 4-layer dimensions; Table II also permits 2/3 layers")
            command.add_argument("--diagnostic", action="store_true",
                help="Permit a short checkpoint/sample budget and label the result diagnostic")
    return parser


def _prepare_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. On this computer use D:/EqDeepRxRuntime/sionna-venv/Scripts/python.exe")
    torch.set_num_threads(4)
    # WDDM can page oversized CUDA reservations into host memory instead
    # of reporting OOM. Cap the allocator itself, leaving room for the
    # driver/display, so unused cached blocks are reclaimed before paging.
    torch.cuda.set_per_process_memory_fraction(.85, torch.cuda.current_device())
    torch.set_float32_matmul_precision("highest")
    # Fixed kernels preserve repeatability across a restarted process.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return torch.device("cuda")


def _new_model(config, device):
    import torch
    from .model import EqDeepRx
    torch.manual_seed(config.training.seed)
    return EqDeepRx(config).to(device, memory_format=torch.channels_last)


def _check_config(config):
    errors = paper_contract_errors(config) + frozen_choice_errors(config)
    if errors:
        raise ValueError("Invalid formal configuration: " + "; ".join(errors))


def _log_event(event, path=None):
    line = json.dumps(event, ensure_ascii=False)
    print(line, flush=True)
    if path is not None:
        with Path(path).open("a", encoding="utf-8") as stream:
            stream.write(line+"\n")


def preflight(config, args, device):
    import torch
    from .evaluation import _atomic_json_write
    from .sionna_system import SionnaTR38901System
    from .training import (config_fingerprint, source_fingerprint,
        paper_training_configurations, train_steps)
    if args.repeats < 2:
        raise ValueError("preflight needs a warmup and at least one measured step")
    _check_config(config)
    run_source_hash = require_source_unchanged()
    system = SionnaTR38901System(config, device)
    cases = []
    total_memory = torch.cuda.get_device_properties(device).total_memory
    for layers, pilots, interference in paper_training_configurations(config):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        events = []
        model = _new_model(config, device)
        train_steps(model, system, config, steps=args.repeats,
            batch_size=config.training.batch_size, microbatch_size=args.microbatch_size,
            generation_batch_size=args.generation_batch_size, n_layers=layers,
            pilot_count=pilots, interference_present=interference, device=device,
            progress_callback=events.append)
        seconds = statistics.median(event["step_seconds"] for event in events[1:])
        record = {"layers": layers, "pilot_count": pilots,
            "interference_present": interference, "steady_step_seconds": seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "last_loss": events[-1]["loss"], "last_ber": events[-1]["ber"],
            "module_optimizer_steps": events[-1]["module_optimizer_steps"]}
        cases.append(record)
        _log_event(record)
        del model
    mean_seconds = 0.0
    for case in cases:
        i = config.layer_counts.index(case["layers"])
        p_layer = config.training.layer_sampling_probabilities[i]
        p_two = config.training.pilot_sampling_probability_two
        p_pilot = p_two if case["pilot_count"] == 2 else 1-p_two
        p_interference = (config.training.interference_probability if case["interference_present"]
            else 1-config.training.interference_probability)
        mean_seconds += p_layer*p_pilot*p_interference*case["steady_step_seconds"]
    memory_safe = max(case["peak_reserved_bytes"] for case in cases) <= .85*total_memory
    require_source_unchanged()
    report = {"long_training_ready": memory_safe,
        "performance_reproduced": False,
        "readiness_scope": "published configuration, joint updates, finite gradients and measured GPU capacity",
        "config_fingerprint": config_fingerprint(config), "source_fingerprint": run_source_hash,
        "torch_version": torch.__version__, "sionna_version": importlib.metadata.version("sionna"),
        "cuda_allocator_config": os.environ.get("PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF")),
        "cuda_memory_fraction": torch.cuda.get_per_process_memory_fraction(torch.cuda.current_device()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "gpu_name": torch.cuda.get_device_name(device), "gpu_total_bytes": total_memory,
        "effective_batch_size": config.training.batch_size,
        "approved_microbatch_size": args.microbatch_size,
        "approved_generation_batch_size": args.generation_batch_size, "sionna_workers": 1,
        "measured_mean_optimizer_step_seconds": mean_seconds,
        "estimated_total_training_days": mean_seconds*config.training.total_steps/86400,
        "estimated_total_training_days_with_15pct_margin": mean_seconds*config.training.total_steps/86400*1.15,
        "cases": cases}
    _atomic_json_write(report, args.run_dir/"preflight.json")
    _log_event({key: value for key, value in report.items() if key != "cases"})
    if not memory_safe:
        raise RuntimeError("preflight peak memory exceeds the reserved 15% margin; reduce computational batch sizes")
    return report


def train(config, args, device):
    import torch
    from .sionna_system import SionnaTR38901System
    from .training import train_steps, validate_full_training_request, source_fingerprint
    _check_config(config)
    report_path = args.run_dir/"preflight.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("torch_version") != torch.__version__ or report.get("sionna_version") != importlib.metadata.version("sionna"):
        raise RuntimeError("runtime dependencies changed since preflight; run preflight again")
    if (report.get("gpu_name") != torch.cuda.get_device_name(device)
            or report.get("gpu_total_bytes") != torch.cuda.get_device_properties(device).total_memory
            or report.get("cuda_allocator_config") != os.environ.get("PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
            or report.get("cuda_memory_fraction") != torch.cuda.get_per_process_memory_fraction(torch.cuda.current_device())
            or report.get("cublas_workspace_config") != os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            or report.get("deterministic_algorithms") != torch.are_deterministic_algorithms_enabled()):
        raise RuntimeError("GPU or memory settings changed since preflight; run preflight again")
    micro, generation = report["approved_microbatch_size"], report["approved_generation_batch_size"]
    validate_full_training_request(config, steps=args.steps,
        batch_size=config.training.batch_size, microbatch_size=micro,
        generation_batch_size=generation, sionna_workers=1, device=device,
        confirm=args.confirm, cuda_available=True, preflight_report=report_path)
    checkpoint = args.run_dir/"eqdeeprx.pt"
    if checkpoint.exists() and not args.resume:
        raise FileExistsError("A checkpoint already exists. Use --resume or a different --run-dir")
    if args.resume and not checkpoint.is_file():
        raise FileNotFoundError("No completed checkpoint is available to resume")
    if args.save_every < 1:
        raise ValueError("save-every must be positive")
    model = _new_model(config, device)
    system = SionnaTR38901System(config, device)
    log_path = args.run_dir/"training.jsonl"
    _log_event({"event": "training_start", "checkpoint": str(checkpoint),
        "resume": args.resume, "target_steps": args.steps,
        "effective_batch_size": config.training.batch_size,
        "source_fingerprint": source_fingerprint()}, log_path)
    try:
        return train_steps(model, system, config, steps=args.steps,
            batch_size=config.training.batch_size, microbatch_size=micro,
            generation_batch_size=generation, device=device, output_path=checkpoint,
            resume_path=checkpoint if args.resume else None, save_every=args.save_every,
            progress_callback=lambda event: _log_event(event, log_path))
    except KeyboardInterrupt:
        _log_event({"event": "interrupted", "resume_from": str(checkpoint),
            "note": "Resume uses the last atomically saved joint model/optimizer snapshot"}, log_path)
        raise


def evaluate(config, args, device):
    import torch
    from .evaluation import evaluate_paper_figure6a, _model_fingerprint
    from .sionna_system import SionnaTR38901System
    from .training import formal_checkpoint_errors, _checkpoint_consistency_errors, Lamb
    path = args.checkpoint or args.run_dir/"eqdeeprx.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved_config = checkpoint["config"]
    _check_config(saved_config)
    signature = checkpoint["run_signature"]
    model = _new_model(saved_config, device)
    if args.diagnostic:
        optimizer = Lamb(model.parameters(), lr=saved_config.training.learning_rate,
            betas=(saved_config.training.lamb_beta1, saved_config.training.lamb_beta2),
            eps=saved_config.training.lamb_eps, weight_decay=saved_config.training.weight_decay,
            bias_correction=saved_config.training.lamb_bias_correction)
        errors = _checkpoint_consistency_errors(checkpoint, model, optimizer,
            expected_config=saved_config, expected_signature=signature,
            expected_amp=checkpoint["amp_enabled"], expected_device=device)
    else:
        if args.samples != saved_config.evaluation.validation_samples:
            raise ValueError("formal Figure 6(a) needs 32000 validation samples per DMRS configuration")
        errors = formal_checkpoint_errors(checkpoint, model, saved_config,
            expected_steps=saved_config.training.total_steps,
            batch_size=saved_config.training.batch_size, microbatch_size=signature["microbatch_size"],
            generation_batch_size=signature["generation_batch_size"], seed=saved_config.training.seed,
            expected_amp=saved_config.training.amp_dtype == "bfloat16", expected_device=device)
    if errors:
        raise ValueError("Invalid checkpoint: "+"; ".join(errors))
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = evaluate_paper_figure6a(model, SionnaTR38901System(saved_config, device), saved_config,
        validation_samples=args.samples, evaluation_batch_size=2, n_layers=args.layers,
        seed=saved_config.training.seed+1_000_000, resume=args.resume,
        diagnostic=args.diagnostic,
        output_dir=args.run_dir/(("diagnostic_figure6a" if args.diagnostic else "figure6a")+f"_{args.layers}layers"),
        checkpoint_provenance={"checkpoint_next_step": checkpoint["next_step"],
            "checkpoint_status": checkpoint["checkpoint_status"],
            "checkpoint_model_fingerprint": _model_fingerprint(model)})
    _log_event({"event": "evaluation_complete", "diagnostic": args.diagnostic,
        "checkpoint_steps": checkpoint["next_step"], "n_layers": args.layers,
        "validation_samples_total": metrics["validation_samples_total"]})
    return metrics


def main():
    args = build_parser().parse_args()
    from .eqdeeprx_runtime import configure_runtime_storage
    configure_runtime_storage(args.run_dir/"runtime")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    device = _prepare_cuda()
    return {"preflight": preflight, "train": train, "evaluate": evaluate}[args.command](paper_config(), args, device)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
