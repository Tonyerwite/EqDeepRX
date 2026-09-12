from __future__ import annotations

import json
import math
import random
import hashlib
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List

import torch

from .config import EqDeepRxConfig
from .losses import (
    ActivationStatistics,
    activation_statistics,
    eqdeeprx_loss,
    vcl_regularization_from_statistics,
)
from .signal import OFDMSystem, SignalBatch, bits_per_symbol


MAX_LONG_TRAINING_DAYS = 8.0
AMP_INITIAL_SCALE = 1.0


def paper_training_configurations(
    config: EqDeepRxConfig,
) -> tuple[tuple[int, int, bool], ...]:
    """Return every layer/pilot/interference case handled by one paper model."""

    return tuple(
        (layers, pilots, interference)
        for layers in config.layer_counts
        for pilots in (1, 2)
        for interference in (False, True)
    )


def config_fingerprint(config: EqDeepRxConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def estimate_training_runtime(
    *,
    smoke_step_seconds: float,
    smoke_batch_size: int,
    effective_batch_size: int,
    total_steps: int,
) -> Dict[str, float]:
    """Linearly extrapolate a measured smoke step to the configured run."""

    if smoke_step_seconds <= 0:
        raise ValueError("smoke_step_seconds must be positive")
    if min(smoke_batch_size, effective_batch_size, total_steps) <= 0:
        raise ValueError("batch sizes and total_steps must be positive")
    seconds_per_sample = smoke_step_seconds / float(smoke_batch_size)
    optimizer_step_seconds = seconds_per_sample * float(effective_batch_size)
    return {
        "samples_per_second": float(smoke_batch_size) / smoke_step_seconds,
        "estimated_optimizer_step_seconds": optimizer_step_seconds,
        "estimated_total_training_days": optimizer_step_seconds
        * float(total_steps)
        / 86_400.0,
    }


def validate_full_training_request(
    config: EqDeepRxConfig,
    *,
    steps: int,
    batch_size: int,
    microbatch_size: int,
    generation_batch_size: int | None = None,
    device: torch.device | str,
    confirm: bool,
    cuda_available: bool,
    preflight_report: Path,
    n_layers: int | None = None,
    pilot_count: int | None = None,
    snr_db: float | None = None,
) -> None:
    """Reject a long run unless every paper-scale preflight condition holds."""

    if steps <= 100:
        return
    if not confirm:
        raise RuntimeError("full training requires explicit confirmation")
    if n_layers is not None:
        raise RuntimeError("full training must sample every paper MIMO layer count")
    if pilot_count is not None:
        raise RuntimeError("full training must sample both paper DMRS configurations")
    if snr_db is not None:
        raise RuntimeError("full training must sample the paper SNR distribution")
    if config.training.backend != "sionna_tr38901":
        raise RuntimeError("full training requires the sionna_tr38901 backend")
    if torch.device(device).type != "cuda" or not cuda_available:
        raise RuntimeError("full training requires an available CUDA device")
    if steps != config.training.total_steps:
        raise RuntimeError(
            f"full training requires exactly {config.training.total_steps} steps"
        )
    if batch_size != config.training.batch_size:
        raise RuntimeError(
            f"full training requires effective batch size {config.training.batch_size}"
        )
    if microbatch_size < 1 or microbatch_size > batch_size:
        raise RuntimeError("full training microbatch size is invalid")
    generation_batch_size = (
        microbatch_size
        if generation_batch_size is None
        else int(generation_batch_size)
    )
    if (
        generation_batch_size < 1
        or generation_batch_size > microbatch_size
        or microbatch_size % generation_batch_size
    ):
        raise RuntimeError("full training generation batch size is invalid")
    path = Path(preflight_report)
    if not path.is_file():
        raise RuntimeError("full training requires a standard preflight report")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("long_training_ready", False):
        raise RuntimeError("standard preflight did not approve long training")
    if report.get("config_fingerprint") != config_fingerprint(config):
        raise RuntimeError("preflight report configuration does not match")
    if report.get("approved_microbatch_size") != microbatch_size:
        raise RuntimeError("full training microbatch size was not approved by preflight")
    if report.get("approved_generation_batch_size") != generation_batch_size:
        raise RuntimeError(
            "full training generation batch size was not approved by preflight"
        )
    runtime_days = report.get("estimated_total_training_days")
    if (
        not isinstance(runtime_days, (int, float))
        or not math.isfinite(runtime_days)
        or runtime_days > MAX_LONG_TRAINING_DAYS
    ):
        raise RuntimeError("standard preflight exceeds the eight-day runtime budget")


class Lamb(torch.optim.Optimizer):
    """LAMB optimizer matching the hyperparameters exposed in the paper run."""

    def __init__(self, params, *, lr: float = 4.4e-3, betas=(0.9, 0.999), eps: float = 1e-6, weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("Lamb does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                grad = parameter.grad
                state["exp_avg"].mul_(beta1).add_(grad, alpha=1.0 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                bias1 = 1.0 - beta1 ** state["step"]
                bias2 = 1.0 - beta2 ** state["step"]
                update = state["exp_avg"] / bias1 / (state["exp_avg_sq"] / bias2).sqrt().add(group["eps"])
                if group["weight_decay"]:
                    update = update + group["weight_decay"] * parameter
                weight_norm = torch.linalg.vector_norm(parameter)
                update_norm = torch.linalg.vector_norm(update)
                if weight_norm == 0 or update_norm == 0 or not torch.isfinite(weight_norm) or not torch.isfinite(update_norm):
                    trust_ratio = 1.0
                else:
                    trust_ratio = float(weight_norm / update_norm)
                parameter.add_(update, alpha=-group["lr"] * trust_ratio)
        return loss


def paper_learning_rate(step: int, *, total_steps: int, base_lr: float, warmup_steps: int = 0) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    if step >= total_steps - 1:
        return 0.0
    decay_start = max(0, warmup_steps)
    progress = float(step - decay_start) / float(max(1, total_steps - decay_start - 1))
    return max(0.0, base_lr * (1.0 - progress))


def _bit_mask(config: EqDeepRxConfig, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(config.model.max_bits, device=device)
    mask[: bits_per_symbol(config.modulation)] = 1.0
    return mask


def compute_ber(logits: torch.Tensor, target_bits: torch.Tensor, data_mask: torch.Tensor, bit_mask: torch.Tensor) -> float:
    if logits.dim() == 4:
        logits = logits.unsqueeze(1)
        target_bits = target_bits.unsqueeze(1)
    if logits.shape[2] != target_bits.shape[2]:
        active_bits = min(logits.shape[2], target_bits.shape[2])
        logits = logits[:, :, :active_bits]
        target_bits = target_bits[:, :, :active_bits]
    if data_mask.dim() == 4:
        data_mask = data_mask.unsqueeze(2)
    elif data_mask.dim() == 3:
        data_mask = data_mask.unsqueeze(1).unsqueeze(2)
    if bit_mask.dim() == 1:
        bit_mask = bit_mask[: logits.shape[2]].view(1, 1, -1, 1, 1)
    elif bit_mask.dim() == 4:
        bit_mask = bit_mask[:, : logits.shape[2]].unsqueeze(1)
    mask = (data_mask * bit_mask).expand_as(logits)
    decisions = (logits < 0).to(target_bits.dtype)
    return float((((decisions != target_bits).to(mask.dtype) * mask).sum() / mask.sum().clamp_min(1.0)).detach().cpu())


def atomic_torch_save(payload: Dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _cached_training_batch(batch: SignalBatch) -> SignalBatch:
    """Keep generated online samples in host memory between VCL passes."""

    def host(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().to("cpu")

    snr = host(batch.snr_db) if isinstance(batch.snr_db, torch.Tensor) else batch.snr_db
    return replace(
        batch,
        received=host(batch.received),
        transmitted=host(batch.transmitted),
        pilot_symbols=host(batch.pilot_symbols),
        pilot_mask=host(batch.pilot_mask),
        data_mask=host(batch.data_mask),
        target_bits=host(batch.target_bits),
        true_channel=torch.empty(0, dtype=batch.true_channel.dtype),
        noise_variance=host(batch.noise_variance),
        snr_db=snr,
    )


def _concatenate_training_batches(parts: List[SignalBatch]) -> SignalBatch:
    if not parts:
        raise ValueError("training batch parts cannot be empty")

    def samples(value: float | torch.Tensor, count: int) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32).flatten()
        if tensor.numel() == 1:
            tensor = tensor.expand(count)
        if tensor.numel() != count:
            raise ValueError("batch metadata must contain one value per sample")
        return tensor

    first = parts[0]
    realized_parts = None
    if all(part.realized_sinr_db is not None for part in parts):
        realized_parts = torch.cat(
            [
                samples(part.realized_sinr_db, part.received.shape[0])
                for part in parts
            ]
        )
    return replace(
        first,
        received=torch.cat([part.received for part in parts]),
        transmitted=torch.cat([part.transmitted for part in parts]),
        pilot_symbols=torch.cat([part.pilot_symbols for part in parts]),
        pilot_mask=torch.cat([part.pilot_mask for part in parts]),
        data_mask=torch.cat([part.data_mask for part in parts]),
        target_bits=torch.cat([part.target_bits for part in parts]),
        true_channel=torch.empty(0, dtype=first.true_channel.dtype),
        noise_variance=torch.cat([part.noise_variance for part in parts]),
        snr_db=torch.cat(
            [samples(part.snr_db, part.received.shape[0]) for part in parts]
        ),
        realized_sinr_db=realized_parts,
    )


def _merge_statistics(parts: List[ActivationStatistics]) -> ActivationStatistics:
    count = sum(part.count for part in parts)
    if count == 0:
        raise ValueError("activation statistics cannot be empty")
    total = sum(part.mean * part.count for part in parts)
    total_squares = sum(
        (part.variance + part.mean.square()) * part.count for part in parts
    )
    mean = total / float(count)
    variance = total_squares / float(count) - mean.square()
    return ActivationStatistics(mean, variance.clamp_min(0.0), count)


def _batch_snr_linear(batch: SignalBatch, device: torch.device) -> torch.Tensor:
    snr_db = torch.as_tensor(batch.snr_db, dtype=torch.float32, device=device)
    return torch.pow(10.0, snr_db / 10.0)


def train_steps(
    model,
    system: OFDMSystem,
    config: EqDeepRxConfig,
    *,
    steps: int,
    batch_size: int | None = None,
    microbatch_size: int | None = None,
    generation_batch_size: int | None = None,
    n_layers: int | None = None,
    pilot_count: int | None = None,
    snr_db: float | None = None,
    seed: int | None = None,
    output_path: Path | None = None,
    resume_path: Path | None = None,
    save_every: int | None = None,
    device: torch.device | str = "cpu",
    use_amp: bool | None = None,
) -> Dict:
    if steps <= 0:
        raise ValueError("steps must be positive")
    device = torch.device(device)
    amp_enabled = device.type == "cuda" if use_amp is None else bool(use_amp)
    if amp_enabled and device.type != "cuda":
        raise ValueError("AMP training is supported only on CUDA")
    model.to(device)
    batch_size = batch_size or config.training.batch_size
    microbatch_size = batch_size if microbatch_size is None else int(microbatch_size)
    if microbatch_size < 1 or microbatch_size > batch_size:
        raise ValueError("microbatch_size must be between 1 and batch_size")
    generation_batch_size = (
        microbatch_size
        if generation_batch_size is None
        else int(generation_batch_size)
    )
    if generation_batch_size < 1 or generation_batch_size > microbatch_size:
        raise ValueError(
            "generation_batch_size must be between 1 and microbatch_size"
        )
    if microbatch_size % generation_batch_size:
        raise ValueError("generation_batch_size must divide microbatch_size")
    seed = config.training.seed if seed is None else int(seed)
    random_state = random.Random(seed)
    if n_layers is not None:
        config.validate_layer_count(n_layers)
    run_signature = {
        "batch_size": batch_size,
        "generation_batch_size": generation_batch_size,
        "microbatch_size": microbatch_size,
        "n_layers": n_layers,
        "pilot_count": pilot_count,
        "seed": seed,
        "snr_db": None if snr_db is None else float(snr_db),
    }
    optimizer = Lamb(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=(config.training.lamb_beta1, config.training.lamb_beta2),
        eps=config.training.lamb_eps,
        weight_decay=config.training.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled, init_scale=AMP_INITIAL_SCALE
    )
    scaler_reset = False
    history = {
        "steps": 0,
        "losses": [],
        "bers": [],
        "learning_rates": [],
        "effective_batch_size": batch_size,
        "microbatch_size": microbatch_size,
        "generation_batch_size": generation_batch_size,
        "amp_enabled": amp_enabled,
    }
    start_step = 0
    if resume_path is not None:
        checkpoint = torch.load(Path(resume_path), map_location=device, weights_only=False)
        saved_config = checkpoint.get("config")
        if (
            saved_config is None
            or config_fingerprint(saved_config) != config_fingerprint(config)
        ):
            raise ValueError("resume checkpoint configuration does not match")
        if checkpoint.get("run_signature") != run_signature:
            raise ValueError("resume checkpoint run parameters do not match")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = checkpoint["history"]
        if history.get("effective_batch_size", batch_size) != batch_size:
            raise ValueError("resume checkpoint effective batch size does not match")
        history["effective_batch_size"] = batch_size
        history["microbatch_size"] = microbatch_size
        if history.get("generation_batch_size", generation_batch_size) != generation_batch_size:
            raise ValueError("resume checkpoint generation batch size does not match")
        history["generation_batch_size"] = generation_batch_size
        saved_amp_enabled = bool(checkpoint.get("amp_enabled", False))
        if saved_amp_enabled != amp_enabled:
            raise ValueError("resume checkpoint AMP setting does not match")
        history["amp_enabled"] = amp_enabled
        if "grad_scaler_state_dict" in checkpoint:
            scaler_state = checkpoint["grad_scaler_state_dict"]
            try:
                saved_scale = float(scaler_state.get("scale", AMP_INITIAL_SCALE))
            except (TypeError, ValueError):
                saved_scale = float("nan")
            if math.isfinite(saved_scale) and saved_scale > 0.0:
                scaler.load_state_dict(scaler_state)
            else:
                # A fully backed-off scaler cannot recover from further steps.
                # Start at the configured scale without changing model math.
                scaler_reset = True
        start_step = int(checkpoint["next_step"])
        if "python_random_state" in checkpoint:
            random_state.setstate(checkpoint["python_random_state"])
        if scaler_reset:
            history["amp_scaler_reset"] = True
    bit_mask = _bit_mask(config, device)
    for step in range(start_step, steps):
        current_layers = n_layers if n_layers is not None else random_state.choice(config.layer_counts)
        config.validate_layer_count(current_layers)
        current_pilot_count = pilot_count if pilot_count is not None else random_state.choice((1, 2))
        add_interference = random_state.random() < config.interference_probability
        batches = []
        batch_parts = []
        model_batch_samples = 0
        generated = 0
        while generated < batch_size:
            current_size = min(generation_batch_size, batch_size - generated)
            current_snr = (
                float(snr_db)
                if snr_db is not None
                else torch.tensor(
                    [
                        random_state.uniform(*config.snr_db_range)
                        for _ in range(current_size)
                    ],
                    dtype=torch.float32,
                )
            )
            batch = system.generate_batch(
                batch_size=current_size,
                n_layers=current_layers,
                pilot_count=current_pilot_count,
                snr_db=current_snr,
                seed=seed + step * batch_size + generated,
                add_interference=add_interference,
                return_true_channel=False,
            )
            batch_parts.append(_cached_training_batch(batch))
            model_batch_samples += current_size
            generated += current_size
            if model_batch_samples == microbatch_size or generated == batch_size:
                batches.append(_concatenate_training_batches(batch_parts))
                batch_parts = []
                model_batch_samples = 0

        model.train()
        statistics_parts: List[List[ActivationStatistics]] = [
            [] for _ in range(config.model.detector_sections)
        ]
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            for batch in batches:
                _, aux = model(
                    batch.received.to(device),
                    batch.pilot_symbols.to(device),
                    batch.pilot_mask.to(device),
                    return_aux=True,
                )
                for section, state in enumerate(aux.get("detector_states", ())):
                    values = state.reshape(
                        -1, state.shape[2], state.shape[3], state.shape[4]
                    )
                    statistics_parts[section].append(
                        activation_statistics((values.detach(),))
                    )
        full_statistics = [_merge_statistics(parts) for parts in statistics_parts]

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_ber = 0.0
        for batch in batches:
            received = batch.received.to(device)
            pilots = batch.pilot_symbols.to(device)
            pilot_mask = batch.pilot_mask.to(device)
            targets = batch.target_bits.to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits, aux = model(received, pilots, pilot_mask, return_aux=True)
                sample_fraction = batch.received.shape[0] / float(batch_size)
                loss = sample_fraction * eqdeeprx_loss(
                    logits,
                    targets,
                    batch.data_mask.to(device),
                    bit_mask,
                    aux["symbol_states"],
                    batch.transmitted.to(device),
                    snr_linear=_batch_snr_linear(batch, device),
                    lambda_symbol=config.training.symbol_loss_weight,
                )
                for section, state in enumerate(aux.get("detector_states", ())):
                    values = state.reshape(
                        -1, state.shape[2], state.shape[3], state.shape[4]
                    )
                    loss = loss + vcl_regularization_from_statistics(
                        values,
                        full_statistics[section],
                        alpha=config.training.vcl_alpha,
                    )
            scaler.scale(loss).backward()
            step_loss += float(loss.detach().cpu())
            step_ber += sample_fraction * compute_ber(
                logits.detach(), targets, batch.data_mask.to(device), bit_mask
            )
        lr = paper_learning_rate(step, total_steps=config.training.total_steps, base_lr=config.training.learning_rate, warmup_steps=config.training.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        scaler.step(optimizer)
        scaler.update()
        history["losses"].append(step_loss)
        history["bers"].append(step_ber)
        history["learning_rates"].append(float(lr))
        history["steps"] = step + 1
        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "config": config,
            "steps": history["steps"],
            "next_step": step + 1,
            "python_random_state": random_state.getstate(),
            "amp_enabled": amp_enabled,
            "grad_scaler_state_dict": scaler.state_dict(),
            "run_signature": run_signature,
        }
        if output_path is not None and (
            step + 1 == steps
            or (save_every is not None and (step + 1) % save_every == 0)
        ):
            atomic_torch_save(payload, Path(output_path))
    return history
