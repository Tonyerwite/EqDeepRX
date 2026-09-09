from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict

import torch

from .config import EqDeepRxConfig
from .losses import eqdeeprx_loss, vcl_regularization
from .signal import OFDMSystem, bits_per_symbol


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
    progress = float(step - decay_start + 1) / float(max(1, total_steps - decay_start))
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
    decisions = (logits > 0).to(target_bits.dtype)
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


def train_steps(
    model,
    system: OFDMSystem,
    config: EqDeepRxConfig,
    *,
    steps: int,
    batch_size: int | None = None,
    n_layers: int | None = None,
    pilot_count: int | None = None,
    snr_db: float | None = None,
    seed: int | None = None,
    output_path: Path | None = None,
    device: torch.device | str = "cpu",
) -> Dict:
    if steps <= 0:
        raise ValueError("steps must be positive")
    device = torch.device(device)
    model.to(device)
    batch_size = batch_size or config.training.batch_size
    seed = config.training.seed if seed is None else int(seed)
    random_state = random.Random(seed)
    if n_layers is not None:
        config.validate_layer_count(n_layers)
    optimizer = Lamb(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=(config.training.lamb_beta1, config.training.lamb_beta2),
        eps=config.training.lamb_eps,
        weight_decay=config.training.weight_decay,
    )
    history = {"steps": 0, "losses": [], "bers": [], "learning_rates": []}
    bit_mask = _bit_mask(config, device)
    for step in range(steps):
        current_layers = n_layers if n_layers is not None else random_state.choice(config.training.layer_counts)
        config.validate_layer_count(current_layers)
        current_pilot_count = pilot_count if pilot_count is not None else random_state.choice((1, 2))
        current_snr = float(snr_db) if snr_db is not None else random_state.uniform(*config.snr_db_range)
        add_interference = random_state.random() < config.training.interference_probability
        batch = system.generate_batch(batch_size=batch_size, n_layers=current_layers, pilot_count=current_pilot_count, snr_db=current_snr, seed=seed + step, add_interference=add_interference)
        received = batch.received.to(device)
        pilots = batch.pilot_symbols.to(device)
        pilot_mask = batch.pilot_mask.to(device)
        targets = batch.target_bits.to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, aux = model(received, pilots, pilot_mask, return_aux=True)
        loss = eqdeeprx_loss(
            logits,
            targets,
            batch.data_mask.to(device),
            bit_mask,
            aux["symbol_states"],
            batch.transmitted.to(device),
            snr_linear=10.0 ** (current_snr / 10.0),
            lambda_symbol=config.training.symbol_loss_weight,
        )
        vcl_loss = loss.new_zeros(())
        for state in aux.get("detector_states", ()):
            # Detector states are [batch, layer, channel, frequency, symbol].
            batch_channel = state.reshape(-1, state.shape[2], state.shape[3], state.shape[4])
            vcl_loss = vcl_loss + vcl_regularization(batch_channel, alpha=config.training.vcl_alpha)
        loss = loss + vcl_loss
        loss.backward()
        lr = paper_learning_rate(step, total_steps=config.training.total_steps, base_lr=config.training.learning_rate, warmup_steps=config.training.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        history["losses"].append(float(loss.detach().cpu()))
        history["bers"].append(compute_ber(logits.detach(), targets, batch.data_mask.to(device), bit_mask))
        history["learning_rates"].append(float(lr))
        history["steps"] = step + 1
    payload = {"model_state_dict": model.state_dict(), "history": history, "config": config, "steps": history["steps"]}
    if output_path is not None:
        atomic_torch_save(payload, Path(output_path))
    return history
