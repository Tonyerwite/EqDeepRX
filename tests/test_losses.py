import torch

from eqdeeprx.losses import eqdeeprx_loss


def test_symbol_loss_reduction_is_explicit_and_changes_only_symbol_term():
    logits = torch.zeros(1, 1, 2, 1, 2)
    targets = torch.zeros_like(logits)
    data_mask = torch.ones(1, 1, 1, 2)
    bit_mask = torch.ones(2)
    state = torch.zeros(1, 1, 2, 1, 2)
    state[:, :, 0] = 1.0
    target_symbols = torch.zeros(1, 1, 1, 2, dtype=torch.complex64)

    summed = eqdeeprx_loss(
        logits,
        targets,
        data_mask,
        bit_mask,
        [state],
        target_symbols,
        lambda_symbol=1.0,
        symbol_reduction="sum",
    )
    averaged = eqdeeprx_loss(
        logits,
        targets,
        data_mask,
        bit_mask,
        [state],
        target_symbols,
        lambda_symbol=1.0,
        symbol_reduction="mean_active",
    )

    assert summed > averaged
    assert torch.isfinite(summed)
    assert torch.isfinite(averaged)
