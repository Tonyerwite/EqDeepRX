from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .config import EqDeepRxConfig
from .receiver import estimate_incm, estimate_raw_channel, interpolate_channel, lmmse_equalize
from .signal import OFDMSystem, bits_per_symbol, qam_demapper_llr
from .training import compute_ber


def _baseline_logits(batch, config: EqDeepRxConfig, *, known_channel: bool) -> torch.Tensor:
    raw = estimate_raw_channel(batch.received, batch.pilot_symbols, batch.pilot_mask)
    channel = batch.true_channel if known_channel else interpolate_channel(raw, batch.pilot_mask)
    covariance = estimate_incm(batch.received, channel, batch.transmitted, batch.pilot_mask, coherence_bandwidth=config.receiver.incm_coherence_bandwidth)
    equalized = lmmse_equalize(batch.received, channel, covariance, coherence_bandwidth=config.receiver.incm_coherence_bandwidth)
    variance = covariance.diagonal(dim1=1, dim2=2).real.mean(dim=(1, 2)).clamp_min(1e-7)
    llrs = []
    for layer in range(equalized.shape[1]):
        llr = qam_demapper_llr(equalized[:, layer], variance, config.modulation, max_bits=config.model.max_bits)
        llrs.append(llr.permute(1, 0, 2, 3))
    return torch.stack(llrs, dim=1)


def _model_ber(model, batch, config: EqDeepRxConfig, bit_mask: torch.Tensor) -> float:
    logits = model(batch.received, batch.pilot_symbols, batch.pilot_mask)
    return compute_ber(logits, batch.target_bits, batch.data_mask, bit_mask)


def evaluate_uncoded_ber(
    model,
    system: OFDMSystem,
    config: EqDeepRxConfig,
    *,
    snr_points: Iterable[float] = (0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0),
    samples_per_point: int = 100,
    n_layers: int = 2,
    seed: int = 2026,
    output_dir: Path | None = None,
) -> Dict:
    if samples_per_point <= 0:
        raise ValueError("samples_per_point must be positive")
    config.validate_layer_count(n_layers)
    snr_values = [float(value) for value in snr_points]
    curves = {"eqdeeprx_1_pilot": [], "eqdeeprx_2_pilots": [], "lmmse_1_pilot": [], "lmmse_2_pilots": [], "lmmse_known_channel": []}
    bit_mask = torch.zeros(config.model.max_bits)
    bit_mask[: bits_per_symbol(config.modulation)] = 1.0
    model.eval()
    with torch.no_grad():
        for snr_index, snr in enumerate(snr_values):
            known_values = []
            for pilot_count in (1, 2):
                eq_values = []
                lmmse_values = []
                for sample in range(samples_per_point):
                    batch = system.generate_batch(batch_size=1, n_layers=n_layers, pilot_count=pilot_count, snr_db=snr, seed=seed + snr_index * 100_000 + sample)
                    eq_values.append(_model_ber(model, batch, config, bit_mask))
                    practical_logits = _baseline_logits(batch, config, known_channel=False)
                    lmmse_values.append(compute_ber(practical_logits, batch.target_bits, batch.data_mask, bit_mask))
                    known_logits = _baseline_logits(batch, config, known_channel=True)
                    known_values.append(compute_ber(known_logits, batch.target_bits, batch.data_mask, bit_mask))
                suffix = "1_pilot" if pilot_count == 1 else "2_pilots"
                curves[f"eqdeeprx_{suffix}"].append(float(sum(eq_values) / len(eq_values)))
                curves[f"lmmse_{suffix}"].append(float(sum(lmmse_values) / len(lmmse_values)))
            curves["lmmse_known_channel"].append(float(sum(known_values) / len(known_values)))
    metrics = {"mode": "eqdeeprx_uncoded_ber", "snr_db": snr_values, "samples_per_point": samples_per_point, "n_layers": n_layers, "seed": seed, "curves": curves}
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "uncoded_ber_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        _plot(metrics, output_dir / "uncoded_ber.png")
    return metrics


def _plot(metrics: Dict, path: Path) -> None:
    plt.figure(figsize=(7.0, 5.0), dpi=160)
    styles = {"eqdeeprx_1_pilot": ("C0", "-o", "EqDeepRx, 1 pilot"), "eqdeeprx_2_pilots": ("C0", "--D", "EqDeepRx, 2 pilots"), "lmmse_1_pilot": ("C3", "-s", "LMMSE, 1 pilot"), "lmmse_2_pilots": ("C3", "--^", "LMMSE, 2 pilots"), "lmmse_known_channel": ("C2", ":X", "LMMSE, known channel")}
    for key, (color, style, label) in styles.items():
        plt.semilogy(metrics["snr_db"], metrics["curves"][key], style, color=color, label=label)
    plt.xlabel("SINR (dB)")
    plt.ylabel("Uncoded BER")
    plt.grid(True, which="both", alpha=0.4)
    plt.legend(loc="lower left")
    plt.ylim(1e-4, 1.0)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

