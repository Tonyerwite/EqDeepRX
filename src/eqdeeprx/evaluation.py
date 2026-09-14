from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .config import EqDeepRxConfig
from .receiver import (
    estimate_incm,
    estimate_raw_channel,
    interpolate_channel,
    lmmse_equalize_with_variance,
    smooth_pilot_channel_frequency,
)
from .signal import OFDMSystem, bits_per_symbol, qam_demapper_llr
from .training import compute_ber


PAPER_FIGURE6A_SINR_POINTS = tuple(
    float(value) for value in range(-5, 14, 2)
)


def _baseline_logits(batch, config: EqDeepRxConfig, *, known_channel: bool) -> torch.Tensor:
    raw = estimate_raw_channel(batch.received, batch.pilot_symbols, batch.pilot_mask)
    channel = (
        batch.true_channel
        if known_channel
        else interpolate_channel(
            smooth_pilot_channel_frequency(
                raw,
                batch.pilot_mask,
                window=config.receiver.baseline_smoothing_window,
            ),
            batch.pilot_mask,
        )
    )
    covariance = estimate_incm(batch.received, channel, batch.transmitted, batch.pilot_mask, coherence_bandwidth=config.receiver.incm_coherence_bandwidth)
    equalized, variance = lmmse_equalize_with_variance(
        batch.received,
        channel,
        covariance,
        coherence_bandwidth=config.receiver.incm_coherence_bandwidth,
    )
    variance = variance.clamp_min(1e-7)
    llrs = []
    for layer in range(equalized.shape[1]):
        llr = qam_demapper_llr(
            equalized[:, layer],
            variance[:, layer],
            config.modulation,
            max_bits=config.model.max_bits,
        )
        llrs.append(llr.permute(1, 0, 2, 3))
    return torch.stack(llrs, dim=1)


def _model_ber(model, batch, config: EqDeepRxConfig, bit_mask: torch.Tensor) -> float:
    logits = model(batch.received, batch.pilot_symbols, batch.pilot_mask)
    return compute_ber(logits, batch.target_bits, batch.data_mask, bit_mask)


def _ber_counts_per_sample(
    logits: torch.Tensor,
    target_bits: torch.Tensor,
    data_mask: torch.Tensor,
    bit_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if logits.dim() == 4:
        logits = logits.unsqueeze(1)
        target_bits = target_bits.unsqueeze(1)
    active_bits = min(logits.shape[2], target_bits.shape[2])
    logits = logits[:, :, :active_bits]
    target_bits = target_bits[:, :, :active_bits].to(logits.device)
    if data_mask.dim() == 4:
        data_mask = data_mask.unsqueeze(2)
    elif data_mask.dim() == 3:
        data_mask = data_mask.unsqueeze(1).unsqueeze(2)
    bit_mask = bit_mask[:active_bits].view(1, 1, -1, 1, 1)
    mask = (data_mask.to(logits.device) * bit_mask).expand_as(logits)
    errors = ((logits < 0).to(target_bits.dtype) != target_bits).to(mask.dtype)
    return (
        (errors * mask).flatten(1).sum(dim=1),
        mask.flatten(1).sum(dim=1),
    )


def _sinr_bin_edges(centers: Iterable[float], width_db: float) -> torch.Tensor:
    values = torch.as_tensor(tuple(float(value) for value in centers))
    if values.numel() == 0:
        raise ValueError("at least one SINR bin center is required")
    if width_db <= 0:
        raise ValueError("SINR bin width must be positive")
    if values.numel() > 1 and not torch.all(values[1:] > values[:-1]):
        raise ValueError("SINR bin centers must be strictly increasing")
    half_width = float(width_db) / 2.0
    return torch.cat((values[:1] - half_width, values + half_width))


def _atomic_json_write(payload: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _new_figure6a_progress(
    curve_names: Iterable[str],
    pilot_counts: Iterable[int],
    n_bins: int,
    signature: Dict,
) -> Dict:
    return {
        "signature": signature,
        "next_sample_by_pilot": {str(pilot): 0 for pilot in pilot_counts},
        "errors": {name: [0.0] * n_bins for name in curve_names},
        "bits": {name: [0.0] * n_bins for name in curve_names},
        "sample_counts": {
            str(pilot): [0] * n_bins for pilot in pilot_counts
        },
    }


def _model_fingerprint(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


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
    metrics = {
        "mode": "fast_smoke_uncoded_ber",
        "backend": "fast_ofdm",
        "paper_figure6a_protocol": False,
        "snr_db": snr_values,
        "samples_per_point": samples_per_point,
        "n_layers": n_layers,
        "seed": seed,
        "curves": curves,
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "uncoded_ber_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        _plot(metrics, output_dir / "uncoded_ber.png")
    return metrics


def evaluate_paper_figure6a(
    model,
    system,
    config: EqDeepRxConfig,
    *,
    sinr_points: Iterable[float] | None = None,
    validation_samples: int | None = None,
    evaluation_batch_size: int = 2,
    n_layers: int = 3,
    seed: int = 2026,
    output_dir: Path | None = None,
    resume: bool = False,
) -> Dict:
    """Evaluate Figure 6(a) using random SNR/INR and realized-SINR bins."""

    validation_samples = (
        config.evaluation.validation_samples
        if validation_samples is None
        else int(validation_samples)
    )
    if validation_samples <= 0:
        raise ValueError("validation_samples must be positive")
    if evaluation_batch_size <= 0:
        raise ValueError("evaluation_batch_size must be positive")
    config.validate_layer_count(n_layers)
    pilot_counts = tuple(config.evaluation.pilot_counts)
    if validation_samples < len(pilot_counts):
        raise ValueError(
            "validation_samples must include at least one sample per pilot configuration"
        )
    samples_per_pilot, remainder = divmod(validation_samples, len(pilot_counts))
    validation_samples_by_pilot = {
        str(pilot): samples_per_pilot + int(index < remainder)
        for index, pilot in enumerate(pilot_counts)
    }
    sinr_centers = [
        float(value)
        for value in (
            PAPER_FIGURE6A_SINR_POINTS
            if sinr_points is None
            else sinr_points
        )
    ]
    curve_names = (
        "eqdeeprx_1_pilot",
        "eqdeeprx_2_pilots",
        "baseline_1_pilot",
        "baseline_2_pilots",
        "baseline_known_channel",
    )
    bin_edges = _sinr_bin_edges(
        sinr_centers, config.evaluation.sinr_bin_width_db
    )
    progress_path = (
        Path(output_dir) / "figure6a_progress.json"
        if output_dir is not None
        else None
    )
    signature = {
        "model": _model_fingerprint(model),
        "sinr_db": sinr_centers,
        "sinr_bin_width_db": config.evaluation.sinr_bin_width_db,
        "validation_samples_total": validation_samples,
        "validation_samples_by_pilot": validation_samples_by_pilot,
        "evaluation_batch_size": evaluation_batch_size,
        "n_layers": n_layers,
        "seed": seed,
        "modulation": config.modulation,
        "channel_model": config.evaluation.channel_model,
        "speed_mps_range": list(config.evaluation.speed_mps_range),
        "snr_db_range": list(config.snr_db_range),
        "pilot_counts": list(pilot_counts),
        "baseline_smoothing_window": config.receiver.baseline_smoothing_window,
    }
    progress = _new_figure6a_progress(
        curve_names,
        pilot_counts,
        len(sinr_centers),
        signature,
    )
    if resume and progress_path is not None and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("signature") != signature:
            raise ValueError(
                "existing Figure 6(a) progress does not match this run"
            )

    bit_mask = torch.zeros(
        config.model.max_bits,
        device=next(model.parameters()).device,
    )
    bit_mask[: bits_per_symbol(config.modulation)] = 1.0
    model.eval()
    with torch.no_grad():
        for pilot_count in pilot_counts:
            pilot_sample_count = validation_samples_by_pilot[str(pilot_count)]
            sample = int(progress["next_sample_by_pilot"][str(pilot_count)])
            while sample < pilot_sample_count:
                current_size = min(
                    evaluation_batch_size, pilot_sample_count - sample
                )
                sampled_snr = torch.tensor(
                    [
                        random.Random(
                            seed + pilot_count * 1_000_000 + index
                        ).uniform(*config.snr_db_range)
                        for index in range(sample, sample + current_size)
                    ],
                    dtype=torch.float32,
                )
                batch = system.generate_batch(
                    batch_size=current_size,
                    n_layers=n_layers,
                    pilot_count=pilot_count,
                    snr_db=sampled_snr,
                    sinr_db=None,
                    seed=seed + pilot_count * 1_000_000 + sample,
                    add_interference=True,
                    channel_model=config.evaluation.channel_model,
                    speed_mps_range=config.evaluation.speed_mps_range,
                )
                if batch.backend != "sionna_tr38901_time_domain":
                    raise RuntimeError(
                        "Figure 6(a) evaluation requires the Sionna time-domain backend"
                    )
                realized = torch.as_tensor(batch.realized_sinr_db).flatten().cpu()
                if realized.numel() != current_size:
                    raise RuntimeError("backend must report one realized SINR per sample")
                bin_indices = torch.bucketize(realized, bin_edges, right=False) - 1
                model_logits = model(
                    batch.received, batch.pilot_symbols, batch.pilot_mask
                )
                baseline_logits = _baseline_logits(batch, config, known_channel=False)
                known_logits = _baseline_logits(batch, config, known_channel=True)
                suffix = "1_pilot" if pilot_count == 1 else "2_pilots"
                batch_counts = {
                    f"eqdeeprx_{suffix}": _ber_counts_per_sample(
                        model_logits,
                        batch.target_bits,
                        batch.data_mask,
                        bit_mask,
                    ),
                    f"baseline_{suffix}": _ber_counts_per_sample(
                        baseline_logits,
                        batch.target_bits,
                        batch.data_mask,
                        bit_mask,
                    ),
                }
                batch_counts["baseline_known_channel"] = _ber_counts_per_sample(
                    known_logits,
                    batch.target_bits,
                    batch.data_mask,
                    bit_mask,
                )
                for local_index, bin_index in enumerate(bin_indices.tolist()):
                    if not 0 <= bin_index < len(sinr_centers):
                        continue
                    progress["sample_counts"][str(pilot_count)][bin_index] += 1
                    for name, (errors, bits) in batch_counts.items():
                        progress["errors"][name][bin_index] += float(errors[local_index])
                        progress["bits"][name][bin_index] += float(bits[local_index])
                sample += current_size
                progress["next_sample_by_pilot"][str(pilot_count)] = sample
                if progress_path is not None:
                    _atomic_json_write(progress, progress_path)

    curves = {
        name: [
            (errors / bits if bits > 0 else None)
            for errors, bits in zip(
                progress["errors"][name], progress["bits"][name]
            )
        ]
        for name in curve_names
    }

    metrics = {
        "mode": "eqdeeprx_figure6a_uncoded_ber",
        "backend": "sionna_tr38901_time_domain",
        "paper_figure6a_protocol": True,
        "bit_identical_to_authors_private_run": False,
        "channel_model": config.evaluation.channel_model,
        "speed_mps_range": list(config.evaluation.speed_mps_range),
        "interfering_ues": config.evaluation.interfering_ues,
        "sinr_db": sinr_centers,
        "sinr_bin_edges_db": bin_edges.tolist(),
        "sinr_bin_width_db": config.evaluation.sinr_bin_width_db,
        "validation_samples_total": validation_samples,
        "validation_samples_by_pilot": validation_samples_by_pilot,
        "known_channel_reference_pilot_count": pilot_counts[0],
        "known_channel_reference_pilot_counts": list(pilot_counts),
        "evaluation_batch_size": evaluation_batch_size,
        "sinr_bin_sample_counts": {
            "1_pilot" if pilot == 1 else "2_pilots": progress["sample_counts"][str(pilot)]
            for pilot in pilot_counts
        },
        "n_layers": n_layers,
        "seed": seed,
        "curves": curves,
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json_write(metrics, output_dir / "figure6a_metrics.json")
        _plot_paper_figure6a(metrics, output_dir / "figure6a_uncoded_ber.png")
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


def _plot_paper_figure6a(metrics: Dict, path: Path) -> None:
    plt.figure(figsize=(7.0, 5.0), dpi=160)
    styles = {
        "eqdeeprx_1_pilot": ("C0", "-o", "EqDeepRx, 1 pilot"),
        "eqdeeprx_2_pilots": ("C0", "--D", "EqDeepRx, 2 pilots"),
        "baseline_1_pilot": ("C3", "-s", "Baseline, 1 pilot"),
        "baseline_2_pilots": ("C3", "--^", "Baseline, 2 pilots"),
        "baseline_known_channel": ("C2", ":X", "Known channel"),
    }
    for key, (color, style, label) in styles.items():
        values = [
            math.nan if value is None else value
            for value in metrics["curves"][key]
        ]
        plt.semilogy(
            metrics["sinr_db"], values, style, color=color, label=label
        )
    plt.xlabel("SINR (dB)")
    plt.ylabel("Uncoded BER")
    plt.grid(True, which="both", alpha=0.4)
    plt.legend(loc="lower left")
    plt.ylim(1e-3, 1.0)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
