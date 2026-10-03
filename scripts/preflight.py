from __future__ import annotations

import argparse
import importlib.metadata
import json
import resource
import sys
import subprocess
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eqdeeprx_runtime import configure_runtime_storage

RUNTIME_ROOT = configure_runtime_storage()

import torch

from eqdeeprx.config import paper_config
from eqdeeprx.model import EqDeepRx
from eqdeeprx.signal import OFDMSystem
from eqdeeprx.training import (
    MAX_LONG_TRAINING_DAYS,
    MAX_MPS_TRAINING_DAYS,
    config_fingerprint,
    estimate_training_runtime,
    paper_training_configurations,
    train_steps,
)

from train import configure_cpu_threads, default_cpu_threads, tiny_config


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def peak_process_rss_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB.
    if sys.platform == "darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def physical_memory_mib() -> float | None:
    try:
        value = int(
            subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return value / (1024.0 * 1024.0)


def swap_usage_text() -> str | None:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "vm.swapusage"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Validate EqDeepRx prerequisites without starting full training.")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--microbatch-size", type=int, default=28)
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=2,
        help="Sionna CPU generation slots per call; 2 is the measured Mac-safe default",
    )
    parser.add_argument(
        "--sionna-workers",
        type=int,
        default=1,
        help="Recorded Sionna worker setting; generation remains synchronous on MPS",
    )
    parser.add_argument(
        "--sionna-cpu-threads",
        type=int,
        default=default_cpu_threads(),
        help="PyTorch CPU threads used by the Sionna generator on macOS",
    )
    parser.add_argument(
        "--output", default=str(RUNTIME_ROOT / "outputs" / "preflight_standard.json")
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    try:
        configure_cpu_threads(args.sionna_cpu_threads)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    config = tiny_config() if args.tiny else paper_config()
    torch.manual_seed(config.training.seed)
    model = EqDeepRx(config)
    parameter_count = model.count_parameters()
    input_elements = config.training.batch_size * config.n_rx_antennas * config.n_subcarriers * config.n_ofdm_symbols
    estimated_input_mib = input_elements * 8 / (1024 ** 2)
    report = {
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": args.device,
        "device_type": torch.device(args.device).type,
        "paper_steps": config.training.total_steps,
        "model_parameters": parameter_count,
        "estimated_received_input_mib": round(estimated_input_mib, 2),
        "full_training_started": False,
        "config_fingerprint": config_fingerprint(config),
        "sionna_cpu_threads": args.sionna_cpu_threads,
        "physical_memory_mib": physical_memory_mib(),
        "swap_usage": swap_usage_text(),
        "long_training_ready": False,
    }
    if args.tiny:
        batch = OFDMSystem(config).generate_batch(batch_size=1, n_layers=2, pilot_count=1, snr_db=12.0, seed=2026)
        with torch.no_grad():
            output = model(batch.received, batch.pilot_symbols, batch.pilot_mask)
        report["tiny_forward_shape"] = list(output.shape)
        report["tiny_forward_finite"] = bool(torch.isfinite(output).all())
    else:
        try:
            from eqdeeprx.sionna_system import SionnaTR38901System

            report["sionna"] = importlib.metadata.version("sionna")
            if torch.device(args.device).type == "cuda":
                torch.cuda.reset_peak_memory_stats(torch.device(args.device))
            elif torch.device(args.device).type == "mps":
                torch.mps.empty_cache()
            # Sionna's time-domain generator is CPU-backed on Apple Silicon.
            # Keep generated batches on CPU and let train_steps transfer only
            # the current model microbatch to MPS.
            system_device = "cpu" if torch.device(args.device).type == "mps" else args.device
            system = SionnaTR38901System(
                config, device=system_device, parallel_workers=args.sionna_workers
            )

            class ObservedSystem:
                def __init__(self, wrapped):
                    self.wrapped = wrapped
                    self.backend = None
                    self.configurations = set()

                def generate_batch(self, **kwargs):
                    batch = self.wrapped.generate_batch(**kwargs)
                    self.backend = batch.backend
                    self.configurations.add(
                        (
                            kwargs["n_layers"],
                            kwargs["pilot_count"],
                            bool(kwargs["add_interference"]),
                        )
                    )
                    return batch

            observed = ObservedSystem(system)
            configurations = paper_training_configurations(config)

            def exercise_configuration(layers, pilots, interference):
                return train_steps(
                    model,
                    observed,
                    config,
                    steps=1,
                    batch_size=args.microbatch_size,
                    microbatch_size=args.microbatch_size,
                    generation_batch_size=args.generation_batch_size,
                    n_layers=layers,
                    pilot_count=pilots,
                    seed=1 if interference else 2,
                    device=args.device,
                )

            for configuration in configurations:
                exercise_configuration(*configuration)
            observed.configurations.clear()
            synchronize(torch.device(args.device))
            started = perf_counter()
            histories = [
                exercise_configuration(*configuration)
                for configuration in configurations
            ]
            synchronize(torch.device(args.device))
            elapsed = perf_counter() - started
            losses = [history["losses"][0] for history in histories]
            gradients_finite = all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
            configuration_coverage = observed.configurations == set(configurations)
            measured_step_seconds = elapsed / float(len(configurations))
            # The 12-case smoke loop uses the requested model microbatch. A
            # separate full effective-batch step captures the online Sionna
            # generation and MPS accumulation cost that linear scaling misses.
            full_batch_started = perf_counter()
            full_batch_history = train_steps(
                model,
                observed,
                config,
                steps=1,
                batch_size=config.training.batch_size,
                microbatch_size=args.microbatch_size,
                generation_batch_size=args.generation_batch_size,
                n_layers=None,
                seed=config.training.seed,
                device=args.device,
            )
            synchronize(torch.device(args.device))
            full_batch_step_seconds = perf_counter() - full_batch_started
            full_batch_loss_finite = bool(
                full_batch_history["losses"]
                and torch.isfinite(torch.tensor(full_batch_history["losses"])).all()
            )
            full_batch_gradients_finite = all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
            report.update(
                {
                    "standard_backend": observed.backend,
                    "warmup_steps": len(configurations),
                    "measured_steps": len(configurations),
                    "smoke_step_seconds": round(measured_step_seconds, 3),
                    "full_batch_step_seconds": round(full_batch_step_seconds, 3),
                    "full_batch_loss_finite": full_batch_loss_finite,
                    "full_batch_gradients_finite": bool(full_batch_gradients_finite),
                    "standard_loss": sum(losses) / len(losses),
                    "standard_loss_finite": bool(
                        torch.isfinite(torch.tensor(losses)).all()
                    ),
                    "standard_gradients_finite": bool(gradients_finite),
                    "configured_effective_batch_size": config.training.batch_size,
                    "smoke_batch_size": args.microbatch_size,
                    "tested_microbatch_size": args.microbatch_size,
                    "tested_generation_batch_size": args.generation_batch_size,
                    "sionna_workers": args.sionna_workers,
                    "configuration_coverage_complete": configuration_coverage,
                    "tested_configurations": [
                        {
                            "layers": layers,
                            "pilot_count": pilots,
                            "interference_present": interference,
                        }
                        for layers, pilots, interference in sorted(
                            observed.configurations
                        )
                    ],
                }
            )
            runtime = estimate_training_runtime(
                smoke_step_seconds=full_batch_step_seconds,
                smoke_batch_size=config.training.batch_size,
                effective_batch_size=config.training.batch_size,
                total_steps=config.training.total_steps,
            )
            report.update({key: round(value, 3) for key, value in runtime.items()})
            memory_safe = False
            report["peak_process_rss_mib"] = round(peak_process_rss_mib(), 1)
            physical_mib = report.get("physical_memory_mib")
            report["process_rss_memory_safe"] = bool(
                physical_mib is not None
                and report["peak_process_rss_mib"] <= 0.85 * float(physical_mib)
            )
            if torch.device(args.device).type == "cuda":
                properties = torch.cuda.get_device_properties(torch.device(args.device))
                peak = torch.cuda.max_memory_allocated(torch.device(args.device))
                report["cuda_device"] = properties.name
                report["cuda_total_memory_mib"] = round(
                    properties.total_memory / 1024**2, 1
                )
                report["cuda_peak_allocated_mib"] = round(peak / 1024**2, 1)
                report["cuda_memory_headroom_mib"] = round(
                    (properties.total_memory - peak) / 1024**2, 1
                )
                memory_safe = peak <= 0.8 * properties.total_memory
                report["cuda_memory_safe"] = bool(memory_safe)
            elif torch.device(args.device).type == "mps":
                recommended = torch.mps.recommended_max_memory()
                driver_allocated = torch.mps.driver_allocated_memory()
                current_allocated = torch.mps.current_allocated_memory()
                report["mps_recommended_max_memory_mib"] = round(
                    recommended / 1024**2, 1
                )
                report["mps_driver_allocated_memory_mib"] = round(
                    driver_allocated / 1024**2, 1
                )
                report["mps_current_allocated_memory_mib"] = round(
                    current_allocated / 1024**2, 1
                )
                report["mps_memory_headroom_mib"] = round(
                    (recommended - driver_allocated) / 1024**2, 1
                )
                memory_safe = driver_allocated <= 0.8 * recommended
                report["mps_memory_safe"] = bool(memory_safe)
                memory_safe = memory_safe and report["process_rss_memory_safe"]
            runtime_limit = (
                MAX_MPS_TRAINING_DAYS
                if torch.device(args.device).type == "mps"
                else MAX_LONG_TRAINING_DAYS
            )
            report["runtime_limit_days"] = runtime_limit
            report["operationally_practical_on_this_device"] = bool(
                report["estimated_total_training_days"] <= runtime_limit
            )
            report["long_training_ready"] = bool(
                torch.device(args.device).type in {"cuda", "mps"}
                and (
                    torch.cuda.is_available()
                    if torch.device(args.device).type == "cuda"
                    else torch.backends.mps.is_available()
                )
                and observed.backend == "sionna_tr38901_time_domain"
                and report["standard_loss_finite"]
                and report["standard_gradients_finite"]
                and report["full_batch_loss_finite"]
                and report["full_batch_gradients_finite"]
                and report["configuration_coverage_complete"]
                and memory_safe
                and report["operationally_practical_on_this_device"]
                and args.microbatch_size >= 1
            )
            report["approved_microbatch_size"] = (
                args.microbatch_size if report["long_training_ready"] else None
            )
            report["approved_generation_batch_size"] = (
                args.generation_batch_size
                if report["long_training_ready"]
                else None
            )
        except Exception as exc:
            report["standard_error"] = f"{type(exc).__name__}: {exc}"
            report["long_training_ready"] = False
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
