from __future__ import annotations

import copy
import json
import math
import random
import hashlib
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List

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
CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_STATUS_COMPLETE = "complete"
CHECKPOINT_STATUS_NONFINITE = "nonfinite"
AMP_DTYPE = torch.bfloat16


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

    def __init__(self, params, *, lr: float = 4.4e-3, betas=(0.9, 0.999), eps: float = 1e-6, weight_decay: float = 0.0, bias_correction: bool = True):
        super().__init__(
            params,
            dict(
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                bias_correction=bool(bias_correction),
            ),
        )

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
                if group["bias_correction"]:
                    bias1 = 1.0 - beta1 ** state["step"]
                    bias2 = 1.0 - beta2 ** state["step"]
                    numerator = state["exp_avg"] / bias1
                    denominator = (state["exp_avg_sq"] / bias2).sqrt()
                else:
                    numerator = state["exp_avg"]
                    denominator = state["exp_avg_sq"].sqrt()
                update = numerator / denominator.add(group["eps"])
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


def _finite_tensor_tree(value: Any, path: str = "value") -> list[str]:
    """Return paths of non-finite tensors/numbers in a nested payload."""

    errors: list[str] = []
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all().item():
            errors.append(path)
    elif isinstance(value, float):
        if not math.isfinite(value):
            errors.append(path)
    elif isinstance(value, dict):
        for key, item in value.items():
            errors.extend(_finite_tensor_tree(item, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            errors.extend(_finite_tensor_tree(item, f"{path}[{index}]"))
    return errors


def _scalar_step(value: Any) -> int | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def _optimizer_hyperparameter_errors(
    saved_groups: Any,
    optimizer: torch.optim.Optimizer,
    *,
    expected_lr: float | None = None,
) -> list[str]:
    """Check the serialized optimizer contract, not just its tensor state."""

    errors: list[str] = []
    if not isinstance(saved_groups, list):
        return ["optimizer hyperparameters: param_groups is malformed"]
    expected_groups = optimizer.param_groups
    if len(saved_groups) != len(expected_groups):
        return [
            "optimizer hyperparameters: param-group count does not match"
        ]
    for index, (saved, expected) in enumerate(zip(saved_groups, expected_groups)):
        if not isinstance(saved, dict):
            errors.append(f"optimizer hyperparameters: group {index} is malformed")
            continue
        for name in ("eps", "weight_decay"):
            try:
                actual_value = float(saved.get(name))
                expected_value = float(expected[name])
            except (TypeError, ValueError, KeyError):
                errors.append(
                    f"optimizer hyperparameters: group {index} {name} is invalid"
                )
                continue
            if not math.isclose(
                actual_value,
                expected_value,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                errors.append(
                    f"optimizer hyperparameters: group {index} {name} "
                    f"{actual_value} != {expected_value}"
                )
        if expected_lr is not None:
            try:
                actual_lr = float(saved["lr"])
            except (KeyError, TypeError, ValueError):
                errors.append(
                    f"learning-rate schedule: group {index} lr is invalid"
                )
            else:
                if not math.isclose(
                    actual_lr,
                    float(expected_lr),
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                ):
                    errors.append(
                        f"learning-rate schedule: group {index} lr "
                        f"{actual_lr} != {float(expected_lr)}"
                    )
        try:
            actual_betas = tuple(float(item) for item in saved["betas"])
            expected_betas = tuple(float(item) for item in expected["betas"])
        except (KeyError, TypeError, ValueError):
            errors.append(
                f"optimizer hyperparameters: group {index} betas are invalid"
            )
        else:
            if actual_betas != expected_betas:
                errors.append(
                    f"optimizer hyperparameters: group {index} betas "
                    f"{actual_betas} != {expected_betas}"
                )
        if bool(saved.get("bias_correction")) != bool(
            expected.get("bias_correction")
        ):
            errors.append(
                f"optimizer hyperparameters: group {index} bias_correction "
                "does not match"
            )
    return errors


def _checkpoint_consistency_errors(
    checkpoint: Dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    expected_config: EqDeepRxConfig,
    expected_signature: Dict[str, Any],
    expected_amp: bool,
) -> list[str]:
    """Validate a resume payload before loading any mutable state."""

    errors: list[str] = []
    status = checkpoint.get("checkpoint_status")
    if status == CHECKPOINT_STATUS_NONFINITE:
        errors.append("checkpoint is marked nonfinite")
    elif status is not None and status != CHECKPOINT_STATUS_COMPLETE:
        errors.append(f"unknown checkpoint status {status!r}")

    try:
        next_step = int(checkpoint["next_step"])
        top_steps = int(checkpoint["steps"])
    except (KeyError, TypeError, ValueError):
        next_step = -1
        top_steps = -2
        errors.append("step metadata is missing or invalid")
    if next_step < 0:
        errors.append("next_step must be non-negative")
    if next_step != top_steps:
        errors.append("step metadata consistency mismatch")

    history = checkpoint.get("history")
    if not isinstance(history, dict):
        errors.append("history consistency: history is missing")
        history = {}
    history_step = _scalar_step(history.get("steps"))
    if history_step != next_step:
        errors.append("history consistency: history.steps does not match next_step")
    for name in ("losses", "bers", "learning_rates"):
        values = history.get(name)
        if not isinstance(values, (list, tuple)) or len(values) != max(next_step, 0):
            errors.append(f"history consistency: {name} length does not match next_step")
        elif _finite_tensor_tree(values, f"history.{name}"):
            errors.append(f"history consistency: {name} contains non-finite values")
    learning_rates = history.get("learning_rates")
    if isinstance(learning_rates, (list, tuple)):
        for index, value in enumerate(learning_rates):
            try:
                actual_lr = float(value)
            except (TypeError, ValueError):
                errors.append(
                    f"learning-rate schedule: history entry {index} is invalid"
                )
                break
            expected_lr = paper_learning_rate(
                index,
                total_steps=expected_config.training.total_steps,
                base_lr=expected_config.training.learning_rate,
                warmup_steps=expected_config.training.warmup_steps,
            )
            if not math.isclose(
                actual_lr,
                expected_lr,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                errors.append(
                    f"learning-rate schedule: history entry {index} "
                    f"{actual_lr} != {expected_lr}"
                )
                break

    saved_config = checkpoint.get("config")
    try:
        config_matches = (
            saved_config is not None
            and config_fingerprint(saved_config) == config_fingerprint(expected_config)
        )
    except (TypeError, ValueError, AttributeError):
        config_matches = False
    if not config_matches:
        errors.append("configuration does not match")
    if checkpoint.get("run_signature") != expected_signature:
        errors.append("run parameters do not match")
    saved_amp = bool(checkpoint.get("amp_enabled", False))
    if saved_amp != expected_amp:
        errors.append("AMP setting does not match")

    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_state, dict):
        errors.append("model state is missing")
    else:
        finite_model_errors = _finite_tensor_tree(model_state, "model_state_dict")
        if finite_model_errors:
            errors.append("model state contains non-finite values")
        expected_keys = set(model.state_dict().keys())
        if set(model_state.keys()) != expected_keys:
            errors.append("model state keys do not match")

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if not isinstance(optimizer_state, dict):
        errors.append("optimizer state consistency: state is missing")
    else:
        state = optimizer_state.get("state")
        groups = optimizer_state.get("param_groups")
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        ids = [parameter_id for group in (groups or []) for parameter_id in group.get("params", [])]
        if not isinstance(state, dict) or not isinstance(groups, list):
            errors.append("optimizer state consistency: malformed state")
        else:
            expected_lr = None
            learning_rates = history.get("learning_rates")
            if isinstance(learning_rates, (list, tuple)) and learning_rates:
                try:
                    expected_lr = float(learning_rates[-1])
                except (TypeError, ValueError):
                    expected_lr = None
            errors.extend(
                _optimizer_hyperparameter_errors(
                    groups,
                    optimizer,
                    expected_lr=expected_lr,
                )
            )
            if len(state) != len(trainable) or len(ids) != len(trainable) or set(state) != set(ids):
                errors.append("optimizer state consistency: one state per trainable parameter is required")
            else:
                for parameter, parameter_id in zip(trainable, ids):
                    entry = state.get(parameter_id)
                    if not isinstance(entry, dict):
                        errors.append(f"optimizer state consistency: missing state {parameter_id}")
                        continue
                    counter = _scalar_step(entry.get("step"))
                    if counter != next_step:
                        errors.append(
                            f"optimizer state consistency: state {parameter_id} step {counter} != {next_step}"
                        )
                    for field in ("exp_avg", "exp_avg_sq"):
                        tensor = entry.get(field)
                        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != tuple(parameter.shape):
                            errors.append(
                                f"optimizer state consistency: state {parameter_id} {field} shape mismatch"
                            )
                    if _finite_tensor_tree(entry, f"optimizer_state_dict.state.{parameter_id}"):
                        errors.append(
                            f"optimizer state consistency: state {parameter_id} contains non-finite values"
                        )

    scaler_state = checkpoint.get("grad_scaler_state_dict")
    if not isinstance(scaler_state, dict):
        errors.append("gradient scaler state is missing")
    elif expected_amp:
        scale = scaler_state.get("scale")
        try:
            scale_value = float(scale)
        except (TypeError, ValueError):
            scale_value = float("nan")
        if not math.isfinite(scale_value) or scale_value <= 0.0:
            errors.append("gradient scaler state is not finite and positive")

    if next_step > 0:
        if "python_random_state" not in checkpoint:
            errors.append("random state is missing")
        if "torch_rng_state" not in checkpoint:
            errors.append("torch RNG state is missing")
        if expected_amp and "cuda_rng_state_all" not in checkpoint:
            errors.append("CUDA RNG state is missing")
    return errors


def formal_checkpoint_errors(
    checkpoint: Dict[str, Any],
    model: torch.nn.Module,
    config: EqDeepRxConfig,
    *,
    expected_steps: int,
    batch_size: int,
    microbatch_size: int,
    generation_batch_size: int,
    seed: int,
    expected_amp: bool,
) -> list[str]:
    """Return contract failures for a checkpoint eligible for formal results."""

    errors: list[str] = []
    if checkpoint.get("checkpoint_status") != CHECKPOINT_STATUS_COMPLETE:
        errors.append("checkpoint is not a complete formal checkpoint")
    if checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        errors.append("checkpoint schema version is not supported")
    next_step = _scalar_step(checkpoint.get("next_step"))
    if next_step != int(expected_steps):
        errors.append(
            f"checkpoint next_step {next_step} does not equal required {expected_steps}"
        )
    expected_signature = {
        "batch_size": int(batch_size),
        "generation_batch_size": int(generation_batch_size),
        "microbatch_size": int(microbatch_size),
        "n_layers": None,
        "pilot_count": None,
        "seed": int(seed),
        "snr_db": None,
    }
    optimizer = Lamb(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=(config.training.lamb_beta1, config.training.lamb_beta2),
        eps=config.training.lamb_eps,
        weight_decay=config.training.weight_decay,
        bias_correction=config.training.lamb_bias_correction,
    )
    errors.extend(
        _checkpoint_consistency_errors(
            checkpoint,
            model,
            optimizer,
            expected_config=config,
            expected_signature=expected_signature,
            expected_amp=expected_amp,
        )
    )
    return errors


def _capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python_random_state": None,
        "torch_rng_state": torch.get_rng_state().clone(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = [item.clone() for item in torch.cuda.get_rng_state_all()]
    return state


def _restore_rng_state(payload: Dict[str, Any], random_state: random.Random) -> None:
    if payload.get("python_random_state") is not None:
        random_state.setstate(payload["python_random_state"])
    if payload.get("torch_rng_state") is not None:
        torch.set_rng_state(payload["torch_rng_state"].cpu())
    if payload.get("cuda_rng_state_all") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [item.to(device="cpu") for item in payload["cuda_rng_state_all"]]
        )


def _runtime_state_errors(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    expected_step: int,
) -> list[str]:
    errors: list[str] = []
    errors.extend(_finite_tensor_tree(model.state_dict(), "model_state_dict"))
    optimizer_state = optimizer.state_dict()
    errors.extend(_finite_tensor_tree(optimizer_state, "optimizer_state_dict"))
    state = optimizer_state.get("state", {})
    groups = optimizer_state.get("param_groups", [])
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    parameter_ids = [parameter_id for group in groups for parameter_id in group.get("params", [])]
    if len(state) != len(trainable) or len(parameter_ids) != len(trainable) or set(state) != set(parameter_ids):
        errors.append("optimizer state does not cover every trainable parameter")
    for parameter_id, entry in state.items():
        if _scalar_step(entry.get("step")) != expected_step:
            errors.append(
                f"optimizer state {parameter_id} step {_scalar_step(entry.get('step'))} != {expected_step}"
            )
    try:
        scale = float(scaler.get_scale())
    except (TypeError, ValueError):
        scale = float("nan")
    if not math.isfinite(scale) or scale <= 0.0:
        errors.append("gradient scaler is non-finite or non-positive")
    return errors


def _copy_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _copy_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def _make_nonfinite_checkpoint_path(output_path: Path, failed_step: int) -> Path:
    output_path = Path(output_path)
    base = output_path.with_name(
        f"{output_path.stem}.nonfinite_step{failed_step}{output_path.suffix}"
    )
    if not base.exists():
        return base
    suffix = 1
    while True:
        candidate = base.with_name(f"{base.stem}_{suffix}{base.suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1


def _write_nonfinite_checkpoint(
    *,
    output_path: Path | None,
    failed_step: int,
    next_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    history: Dict[str, Any],
    config: EqDeepRxConfig,
    run_signature: Dict[str, Any],
    amp_enabled: bool,
    scaler: torch.amp.GradScaler,
    diagnostics: list[str],
    python_random_state: object | None = None,
) -> None:
    if output_path is None:
        return
    rng = _capture_rng_state()
    payload = {
        "checkpoint_status": CHECKPOINT_STATUS_NONFINITE,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "failed_step": int(failed_step),
        "next_step": int(next_step),
        "steps": int(next_step),
        "diagnostics": list(diagnostics),
        "model_state_dict": _copy_to_cpu(model.state_dict()),
        "optimizer_state_dict": _copy_to_cpu(optimizer.state_dict()),
        "history": _copy_to_cpu(history),
        "config": config,
        "run_signature": copy.deepcopy(run_signature),
        "amp_enabled": amp_enabled,
        "grad_scaler_state_dict": _copy_to_cpu(scaler.state_dict()),
        **rng,
    }
    if python_random_state is not None:
        payload["python_random_state"] = copy.deepcopy(python_random_state)
    atomic_torch_save(payload, _make_nonfinite_checkpoint_path(Path(output_path), failed_step))


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
    amp_dtype = AMP_DTYPE if config.training.amp_dtype == "bfloat16" else torch.float32
    amp_enabled = (
        device.type == "cuda" and amp_dtype != torch.float32
        if use_amp is None
        else bool(use_amp) and device.type == "cuda" and amp_dtype != torch.float32
    )
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
        bias_correction=config.training.lamb_bias_correction,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled, init_scale=AMP_INITIAL_SCALE
    )
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
        consistency_errors = _checkpoint_consistency_errors(
            checkpoint,
            model,
            optimizer,
            expected_config=config,
            expected_signature=run_signature,
            expected_amp=amp_enabled,
        )
        if consistency_errors:
            raise ValueError(
                "invalid resume checkpoint: " + "; ".join(consistency_errors)
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = copy.deepcopy(checkpoint["history"])
        if history.get("effective_batch_size", batch_size) != batch_size:
            raise ValueError("resume checkpoint effective batch size does not match")
        if history.get("generation_batch_size", generation_batch_size) != generation_batch_size:
            raise ValueError("resume checkpoint generation batch size does not match")
        history["effective_batch_size"] = batch_size
        history["microbatch_size"] = microbatch_size
        history["generation_batch_size"] = generation_batch_size
        history["amp_enabled"] = amp_enabled
        scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
        start_step = int(checkpoint["next_step"])
        _restore_rng_state(checkpoint, random_state)
    bit_mask = _bit_mask(config, device)
    for step in range(start_step, steps):
        def fail_nonfinite(diagnostics: list[str]) -> None:
            _write_nonfinite_checkpoint(
                output_path=output_path,
                failed_step=step,
                next_step=step,
                model=model,
                optimizer=optimizer,
                history=history,
                config=config,
                run_signature=run_signature,
                amp_enabled=amp_enabled,
                scaler=scaler,
                diagnostics=diagnostics,
                python_random_state=random_state.getstate(),
            )
            raise FloatingPointError(
                f"non-finite training state at step {step}: {', '.join(diagnostics)}"
            )

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
        selected_sections = set(range(config.model.detector_sections))
        if config.training.vcl_attachment == "final":
            selected_sections = {config.model.detector_sections - 1}
        elif config.training.vcl_attachment == "none":
            selected_sections = set()
        if selected_sections:
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
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
                        if section not in selected_sections:
                            continue
                        values = state.reshape(
                            -1, state.shape[2], state.shape[3], state.shape[4]
                        )
                        statistics_parts[section].append(
                            activation_statistics((values.detach(),))
                        )
        full_statistics = [
            _merge_statistics(parts) if section in selected_sections else None
            for section, parts in enumerate(statistics_parts)
        ]
        statistics_errors = _finite_tensor_tree(full_statistics, "activation_statistics")
        if statistics_errors:
            fail_nonfinite(statistics_errors)

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
                dtype=amp_dtype,
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
                    symbol_reduction=config.training.symbol_loss_reduction,
                )
                for section, state in enumerate(aux.get("detector_states", ())):
                    if section not in selected_sections:
                        continue
                    values = state.reshape(
                        -1, state.shape[2], state.shape[3], state.shape[4]
                    )
                    loss = loss + vcl_regularization_from_statistics(
                        values,
                        full_statistics[section],
                        alpha=config.training.vcl_alpha,
                    )
                if not torch.isfinite(loss.detach()).all().item():
                    fail_nonfinite([f"loss.microbatch{len(batch_parts)}"])
            scaler.scale(loss).backward()
            step_loss += float(loss.detach().cpu())
            step_ber += sample_fraction * compute_ber(
                logits.detach(), targets, batch.data_mask.to(device), bit_mask
            )
        if not math.isfinite(step_loss) or not math.isfinite(step_ber):
            fail_nonfinite(["history.losses" if not math.isfinite(step_loss) else "history.bers"])
        if amp_enabled:
            scaler.unscale_(optimizer)
        gradient_errors: list[str] = []
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                gradient_errors.append(f"gradient.{name}.missing")
            elif not torch.isfinite(parameter.grad).all().item():
                gradient_errors.append(f"gradient.{name}")
        if gradient_errors:
            fail_nonfinite(gradient_errors)
        lr = paper_learning_rate(step, total_steps=config.training.total_steps, base_lr=config.training.learning_rate, warmup_steps=config.training.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        scaler.step(optimizer)
        scaler.update()
        proposed_step = step + 1
        runtime_errors = _runtime_state_errors(
            model,
            optimizer,
            scaler,
            expected_step=proposed_step,
        )
        if runtime_errors:
            fail_nonfinite(runtime_errors)
        history["losses"].append(step_loss)
        history["bers"].append(step_ber)
        history["learning_rates"].append(float(lr))
        history["steps"] = proposed_step
        rng = _capture_rng_state()
        rng["python_random_state"] = random_state.getstate()
        payload = {
            "checkpoint_status": CHECKPOINT_STATUS_COMPLETE,
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "config": config,
            "steps": history["steps"],
            "next_step": proposed_step,
            "amp_enabled": amp_enabled,
            "grad_scaler_state_dict": scaler.state_dict(),
            "run_signature": run_signature,
            **rng,
        }
        if output_path is not None and (
            proposed_step == steps
            or (save_every is not None and proposed_step % save_every == 0)
        ):
            atomic_torch_save(payload, Path(output_path))
    return history
