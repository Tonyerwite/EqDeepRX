from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eqdeeprx.config import paper_config
from eqdeeprx.evaluation import evaluate_uncoded_ber
from eqdeeprx.model import EqDeepRx
from eqdeeprx.signal import OFDMSystem

from train import tiny_config


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate EqDeepRx and LMMSE uncoded BER.")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output-dir", default="outputs/uncoded_ber")
    parser.add_argument("--snr-points", default="0,3,6,9,12,15,18,21")
    parser.add_argument("--samples-per-point", type=int, default=100)
    parser.add_argument("--n-layers", type=int, default=2, choices=(2, 3, 4))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tiny", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    config = tiny_config() if args.tiny else paper_config()
    model = EqDeepRx(config).to(args.device)
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        model.load_state_dict(payload.get("model_state_dict", payload))
    snr_points = [float(item.strip()) for item in args.snr_points.split(",") if item.strip()]
    metrics = evaluate_uncoded_ber(model, OFDMSystem(config), config, snr_points=snr_points, samples_per_point=args.samples_per_point, n_layers=2 if args.tiny else args.n_layers, seed=args.seed, output_dir=Path(args.output_dir))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

