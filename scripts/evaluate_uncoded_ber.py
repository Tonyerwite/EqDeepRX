from __future__ import annotations

import argparse
import json
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
from eqdeeprx.evaluation import evaluate_paper_figure6a, evaluate_uncoded_ber
from eqdeeprx.model import EqDeepRx
from eqdeeprx.signal import OFDMSystem
from eqdeeprx.training import config_fingerprint

from train import tiny_config


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate EqDeepRx and LMMSE uncoded BER.")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument(
        "--output-dir", default=str(RUNTIME_ROOT / "outputs" / "uncoded_ber")
    )
    parser.add_argument(
        "--sinr-points",
        "--snr-points",
        dest="sinr_points",
        default="-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10",
    )
    parser.add_argument("--validation-samples", type=int, default=None)
    parser.add_argument("--samples-per-point", type=int, default=None)
    parser.add_argument("--evaluation-batch-size", type=int, default=2)
    parser.add_argument("--n-layers", type=int, default=4, choices=(2, 3, 4))
    parser.add_argument(
        "--modulation", default=None, choices=("16QAM", "64QAM", "256QAM")
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tiny", action="store_true")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if not args.tiny and not args.checkpoint:
        parser.error("--checkpoint is required for standard Figure 6(a) evaluation")
    config = tiny_config() if args.tiny else paper_config()
    if args.modulation is not None:
        config = config.with_modulation(args.modulation)
    model = EqDeepRx(config).to(args.device)
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        if not args.tiny:
            saved_config = payload.get("config")
            if (
                saved_config is None
                or config_fingerprint(saved_config) != config_fingerprint(config)
            ):
                raise SystemExit("checkpoint configuration does not match evaluation")
        model.load_state_dict(payload.get("model_state_dict", payload))
    points = [float(item.strip()) for item in args.sinr_points.split(",") if item.strip()]
    if args.tiny:
        samples_per_point = args.samples_per_point or args.validation_samples or 100
        metrics = evaluate_uncoded_ber(
            model,
            OFDMSystem(config, device=args.device),
            config,
            snr_points=points,
            samples_per_point=samples_per_point,
            n_layers=2,
            seed=args.seed,
            output_dir=Path(args.output_dir),
        )
    else:
        from eqdeeprx.sionna_system import SionnaTR38901System

        metrics = evaluate_paper_figure6a(
            model,
            SionnaTR38901System(config, device=args.device),
            config,
            sinr_points=points,
            validation_samples=args.validation_samples,
            evaluation_batch_size=args.evaluation_batch_size,
            n_layers=args.n_layers,
            seed=args.seed,
            output_dir=Path(args.output_dir),
            resume=args.resume,
        )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
