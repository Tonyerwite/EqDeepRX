from dataclasses import replace

import torch

from eqdeeprx.config import paper_config
from eqdeeprx.losses import eqdeeprx_loss
from eqdeeprx.losses import vcl_regularization
from eqdeeprx.signal import OFDMSystem
from eqdeeprx.training import Lamb, paper_learning_rate, train_steps
from eqdeeprx.model import EqDeepRx


def _tiny_config():
    return replace(
        paper_config().with_modulation("16QAM"),
        n_subcarriers=16,
        n_fft=24,
        cyclic_prefix=4,
        n_rx_antennas=2,
        n_tx_antennas=2,
        layer_counts=(2,),
        model=replace(paper_config().model, detector_channels=8, detector_sections=1, demapper_widths=(4, 4, 4, 4), denoise_widths=(8, 8, 8, 2), denoise_subsamples=(1, 2, 2, 1)),
    )


def test_loss_uses_positive_logits_and_snr_weighted_symbol_term():
    target = torch.tensor([[[[[0.0, 1.0], [1.0, 0.0]]]]])
    logits = (target * 2.0 - 1.0) * 8.0
    mask = torch.ones(1, 1, 2, 2)
    bits = torch.ones(1, 1, 1, 1)
    state = [torch.zeros(1, 1, 2, 2, 2)]
    symbols = torch.ones(1, 1, 2, 2, dtype=torch.cfloat)

    low = eqdeeprx_loss(logits, target, mask, bits, state, symbols, snr_linear=1.0, lambda_symbol=1e-2)
    high = eqdeeprx_loss(logits, target, mask, bits, state, symbols, snr_linear=15.0, lambda_symbol=1e-2)
    inverted = eqdeeprx_loss(-logits, target, mask, bits, [], None, snr_linear=1.0, lambda_symbol=0.0)

    assert low.item() < high.item()
    assert inverted.item() > low.item()


def test_vcl_regularization_penalizes_nonzero_mean_and_nonunit_variance():
    activations = torch.ones(2, 3, 4, 4)
    assert vcl_regularization(activations).item() > 0.0
    assert vcl_regularization(torch.randn(2, 3, 4, 4)).item() >= 0.0


def test_paper_lamb_and_linear_schedule_contract():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = Lamb([parameter], lr=4.4e-3, betas=(0.9, 0.999), eps=1e-6)
    before = parameter.detach().clone()
    (parameter.square()).backward()
    optimizer.step()
    assert not torch.equal(before, parameter.detach())
    assert paper_learning_rate(0, total_steps=10, base_lr=1.0, warmup_steps=0) == 0.9
    assert paper_learning_rate(4, total_steps=10, base_lr=1.0, warmup_steps=0) == 0.5
    assert paper_learning_rate(9, total_steps=10, base_lr=1.0, warmup_steps=0) == 0.0


def test_short_training_is_deterministic_and_writes_atomic_checkpoint(tmp_path):
    config = _tiny_config()
    torch.manual_seed(21)
    first_model = EqDeepRx(config)
    torch.manual_seed(21)
    second_model = EqDeepRx(config)
    first = train_steps(first_model, OFDMSystem(config), config, steps=1, batch_size=1, n_layers=2, pilot_count=1, snr_db=12.0, seed=12, output_path=tmp_path / "first.pt")
    second = train_steps(second_model, OFDMSystem(config), config, steps=1, batch_size=1, n_layers=2, pilot_count=1, snr_db=12.0, seed=12, output_path=tmp_path / "second.pt")

    assert first["steps"] == second["steps"] == 1
    assert first["losses"] == second["losses"]
    assert (tmp_path / "first.pt").exists()
    assert not (tmp_path / "first.pt.tmp").exists()
