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
from eqdeeprx.model import EqDeepRx
from eqdeeprx.signal import OFDMSystem

from train import tiny_config


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Validate EqDeepRx prerequisites without starting full training.")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main():
    args = build_arg_parser().parse_args()
    config = tiny_config() if args.tiny else paper_config()
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
    }
    if args.tiny:
        batch = OFDMSystem(config).generate_batch(batch_size=1, n_layers=2, pilot_count=1, snr_db=12.0, seed=2026)
        with torch.no_grad():
            output = model(batch.received, batch.pilot_symbols, batch.pilot_mask)
        report["tiny_forward_shape"] = list(output.shape)
        report["tiny_forward_finite"] = bool(torch.isfinite(output).all())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

