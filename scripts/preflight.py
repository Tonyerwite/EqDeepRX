from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
from time import perf_counter

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eqdeeprx.config import paper_config
from eqdeeprx.model import EqDeepRx
from eqdeeprx.signal import OFDMSystem
from eqdeeprx.training import (
    MAX_LONG_TRAINING_DAYS,
    config_fingerprint,
    estimate_training_runtime,
    paper_training_configurations,
    train_steps,
)

from train import tiny_config


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Validate EqDeepRx prerequisites without starting full training.")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--microbatch-size", type=int, default=28)
    parser.add_argument("--generation-batch-size", type=int, default=2)
    parser.add_argument("--output", default="outputs/preflight_standard.json")
    return parser


def main():
    args = build_arg_parser().parse_args()
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
        "paper_steps": config.training.total_steps,
        "model_parameters": parameter_count,
        "estimated_received_input_mib": round(estimated_input_mib, 2),
        "full_training_started": False,
        "config_fingerprint": config_fingerprint(config),
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
            system = SionnaTR38901System(config, device=args.device)

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
            if torch.device(args.device).type == "cuda":
                torch.cuda.synchronize(torch.device(args.device))
            started = perf_counter()
            histories = [
                exercise_configuration(*configuration)
                for configuration in configurations
            ]
            if torch.device(args.device).type == "cuda":
                torch.cuda.synchronize(torch.device(args.device))
            elapsed = perf_counter() - started
            losses = [history["losses"][0] for history in histories]
            gradients_finite = all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
            configuration_coverage = observed.configurations == set(configurations)
            measured_step_seconds = elapsed / float(len(configurations))
            report.update(
                {
                    "standard_backend": observed.backend,
                    "warmup_steps": len(configurations),
                    "measured_steps": len(configurations),
                    "smoke_step_seconds": round(measured_step_seconds, 3),
                    "standard_loss": sum(losses) / len(losses),
                    "standard_loss_finite": bool(
                        torch.isfinite(torch.tensor(losses)).all()
                    ),
                    "standard_gradients_finite": bool(gradients_finite),
                    "configured_effective_batch_size": config.training.batch_size,
                    "smoke_batch_size": args.microbatch_size,
                    "tested_microbatch_size": args.microbatch_size,
                    "tested_generation_batch_size": args.generation_batch_size,
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
                smoke_step_seconds=measured_step_seconds,
                smoke_batch_size=args.microbatch_size,
                effective_batch_size=config.training.batch_size,
                total_steps=config.training.total_steps,
            )
            report.update({key: round(value, 3) for key, value in runtime.items()})
            memory_safe = False
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
            report["operationally_practical_on_this_device"] = bool(
                report["estimated_total_training_days"] <= MAX_LONG_TRAINING_DAYS
            )
            report["long_training_ready"] = bool(
                torch.device(args.device).type == "cuda"
                and torch.cuda.is_available()
                and observed.backend == "sionna_tr38901_time_domain"
                and report["standard_loss_finite"]
                and report["standard_gradients_finite"]
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
