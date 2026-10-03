from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

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
from eqdeeprx.training import train_steps, validate_full_training_request


def tiny_config():
    config = paper_config().with_modulation("16QAM")
    return replace(config, n_subcarriers=16, n_fft=24, cyclic_prefix=4, n_rx_antennas=2, n_tx_antennas=2, layer_counts=(2,), training=replace(config.training, layer_counts=(2,), interference_probability=0.0), model=replace(config.model, detector_channels=8, detector_sections=1, demapper_widths=(4, 4, 4, 4), denoise_widths=(8, 8, 8, 2), denoise_subsamples=(1, 2, 2, 1)))


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_cpu_threads() -> int:
    """Use the available Apple CPU without oversubscribing the host."""

    return max(1, min(12, int(os.cpu_count() or 1)))


def configure_cpu_threads(threads: int) -> None:
    if int(threads) < 1:
        raise ValueError("sionna CPU thread count must be positive")
    value = str(int(threads))
    # These environment variables affect libraries initialized after import;
    # the explicit PyTorch settings cover the already-loaded runtime.
    os.environ.setdefault("OMP_NUM_THREADS", value)
    os.environ.setdefault("MKL_NUM_THREADS", value)
    torch.set_num_threads(int(threads))
    try:
        torch.set_num_interop_threads(int(threads))
    except RuntimeError:
        # A library may have initialized the inter-op pool before the CLI
        # starts. Intra-op parallelism is still configured above.
        pass


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train EqDeepRx up to uncoded BER.")
    parser.add_argument("--steps", type=int, default=70_000)
    parser.add_argument("--batch-size", type=int, default=112)
    parser.add_argument("--microbatch-size", type=int, default=28)
    # Apple Silicon's Sionna CPU generator is fastest and memory-safe at two
    # generated slots per call for the paper-scale effective batch.
    parser.add_argument("--generation-batch-size", type=int, default=2)
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
    parser.add_argument("--n-layers", type=int, default=None, choices=(2, 3, 4))
    parser.add_argument("--pilot-count", type=int, default=None, choices=(1, 2))
    parser.add_argument(
        "--modulation", default=None, choices=("16QAM", "64QAM", "256QAM")
    )
    parser.add_argument("--snr-db", type=float, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default=default_device())
    parser.add_argument(
        "--output", default=str(RUNTIME_ROOT / "checkpoints" / "eqdeeprx.pt")
    )
    parser.add_argument("--resume", default="")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument(
        "--preflight-report",
        default=str(RUNTIME_ROOT / "outputs" / "preflight_standard.json"),
    )
    parser.add_argument("--tiny", action="store_true", help="Use a CPU-sized 16-subcarrier smoke configuration.")
    parser.add_argument("--confirm-full-run", action="store_true", help="Required for paper-scale runs over 100 steps.")
    return parser


def main():
    args = build_arg_parser().parse_args()
    try:
        configure_cpu_threads(args.sionna_cpu_threads)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    config = tiny_config() if args.tiny else paper_config()
    if args.modulation is not None:
        config = config.with_modulation(args.modulation)
    if args.tiny:
        args.steps = min(args.steps, 1)
        args.batch_size = min(args.batch_size, 2)
        args.microbatch_size = min(args.microbatch_size, args.batch_size)
        args.generation_batch_size = min(
            args.generation_batch_size, args.microbatch_size
        )
        system = OFDMSystem(config, device=args.device)
    else:
        try:
            validate_full_training_request(
                config,
                steps=args.steps,
                batch_size=args.batch_size,
                microbatch_size=args.microbatch_size,
                generation_batch_size=args.generation_batch_size,
                sionna_workers=args.sionna_workers,
                device=args.device,
                confirm=args.confirm_full_run,
                cuda_available=torch.cuda.is_available(),
                mps_available=torch.backends.mps.is_available(),
                preflight_report=Path(args.preflight_report),
                n_layers=args.n_layers,
                pilot_count=args.pilot_count,
                snr_db=args.snr_db,
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        from eqdeeprx.sionna_system import SionnaTR38901System

        system_device = "cpu" if torch.device(args.device).type == "mps" else args.device
        system = SionnaTR38901System(
            config, device=system_device, parallel_workers=args.sionna_workers
        )
    torch.manual_seed(args.seed)
    model = EqDeepRx(config)
    result = train_steps(
        model,
        system,
        config,
        steps=args.steps,
        batch_size=args.batch_size,
        microbatch_size=args.microbatch_size,
        generation_batch_size=args.generation_batch_size,
        n_layers=args.n_layers if not args.tiny else 2,
        pilot_count=args.pilot_count,
        snr_db=args.snr_db,
        seed=args.seed,
        output_path=Path(args.output),
        resume_path=Path(args.resume) if args.resume else None,
        save_every=args.save_every,
        device=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
