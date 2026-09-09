from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn.functional as F


def vcl_regularization(
    activations: torch.Tensor,
    *,
    alpha: float = 1e-5,
    channel_dim: int = 1,
    target_mean: float = 0.0,
    target_variance: float = 1.0,
) -> torch.Tensor:
    """Compute the paper's per-channel mean/variance VCL regularizer."""

    if activations.ndim < 2:
        raise ValueError("activations must have a batch and channel dimension")
    if channel_dim < 0:
        channel_dim += activations.ndim
    if not 0 <= channel_dim < activations.ndim:
        raise ValueError("channel_dim is outside the activation tensor")
    if activations.numel() == 0:
        return activations.new_zeros(())
    values = activations.movedim(channel_dim, 1)
    reduce_dims = (0,) + tuple(range(2, values.ndim))
    means = values.mean(dim=reduce_dims)
    variances = values.var(dim=reduce_dims, unbiased=False)
    mean_penalty = (means - float(target_mean)).square().mean()
    variance_penalty = (variances - float(target_variance)).square().mean()
    return float(alpha) * mean_penalty + variance_penalty


def _mask_for_logits(mask: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=logits.device, dtype=logits.dtype)
    if logits.dim() == 5:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1).unsqueeze(2)
        elif mask.dim() == 4:
            mask = mask.unsqueeze(2)
        elif mask.dim() != 5:
            raise ValueError("data_mask must have 3, 4, or 5 dimensions")
    else:
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
    return mask


def _bits_for_logits(bit_mask: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    mask = bit_mask.to(device=logits.device, dtype=logits.dtype)
    if logits.dim() == 5:
        if mask.dim() == 1:
            return mask[: logits.shape[2]].view(1, 1, -1, 1, 1)
        if mask.dim() == 4:
            return mask[:, :, : logits.shape[2]].unsqueeze(1)
        return mask
    if mask.dim() == 1:
        return mask[: logits.shape[1]].view(1, -1, 1, 1)
    if mask.dim() == 3:
        return mask.unsqueeze(0)
    return mask


def masked_bce(logits: torch.Tensor, target_bits: torch.Tensor, data_mask: torch.Tensor, bit_mask: torch.Tensor) -> torch.Tensor:
    full_mask = (_mask_for_logits(data_mask, logits) * _bits_for_logits(bit_mask, logits)).expand_as(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target_bits.to(logits.device).to(logits.dtype), reduction="none")
    return (bce * full_mask).sum() / full_mask.sum().clamp_min(1.0)


def eqdeeprx_loss(
    logits: torch.Tensor,
    target_bits: torch.Tensor,
    data_mask: torch.Tensor,
    bit_mask: torch.Tensor,
    symbol_states: Sequence[torch.Tensor] | None = None,
    target_symbols: torch.Tensor | None = None,
    *,
    snr_linear: torch.Tensor | float = 1.0,
    lambda_symbol: float = 1e-5,
) -> torch.Tensor:
    """Equation (13), with positive logits denoting bit one."""

    if logits.shape[-3] != target_bits.shape[-3]:
        active_bits = min(logits.shape[-3], target_bits.shape[-3])
        logits = logits.narrow(-3, 0, active_bits)
        target_bits = target_bits.narrow(-3, 0, active_bits)

    mask = _mask_for_logits(data_mask, logits)
    bits = _bits_for_logits(bit_mask, logits)
    full_mask = (mask * bits).expand_as(logits)
    per_element = F.binary_cross_entropy_with_logits(logits, target_bits.to(logits.device).to(logits.dtype), reduction="none")
    per_sample = (per_element * full_mask).reshape(logits.shape[0], -1).sum(dim=1) / full_mask.reshape(logits.shape[0], -1).sum(dim=1).clamp_min(1.0)

    symbol_loss = logits.new_zeros(logits.shape[0])
    if symbol_states and target_symbols is not None:
        target = target_symbols.to(logits.device)
        symbol_mask = mask
        if logits.dim() == 5:
            symbol_mask = mask[:, :, 0]
        else:
            symbol_mask = mask[:, 0, 0]
        for state in symbol_states:
            state = state.to(logits.device)
            if state.dim() != 5:
                raise ValueError("symbol states must be [N,L,2,F,S]")
            prediction = torch.complex(state[:, :, 0], state[:, :, 1])
            error = (prediction - target).abs().square() * symbol_mask
            symbol_loss = symbol_loss + error.reshape(error.shape[0], -1).sum(dim=1) / symbol_mask.reshape(symbol_mask.shape[0], -1).sum(dim=1).clamp_min(1.0)

    weight = torch.as_tensor(snr_linear, device=logits.device, dtype=logits.dtype)
    if weight.dim() == 0:
        weight = weight.expand(logits.shape[0])
    weight = torch.log2(1.0 + weight.clamp_min(0.0)).reshape(-1)
    if weight.numel() != logits.shape[0]:
        raise ValueError("snr_linear must be scalar or have one value per batch sample")
    return (weight * (per_sample + float(lambda_symbol) * symbol_loss)).mean()
