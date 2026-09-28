from dataclasses import replace

import torch

from eqdeeprx.config import paper_config
from eqdeeprx.model import EqDeepRx
from eqdeeprx.signal import OFDMSystem
from eqdeeprx.training import train_steps
from scripts.audit_checkpoints import audit_payload


def _tiny_config():
    base = paper_config().with_modulation("16QAM")
    return replace(
        base,
        n_subcarriers=16,
        n_fft=24,
        cyclic_prefix=4,
        n_rx_antennas=2,
        n_tx_antennas=2,
        layer_counts=(2,),
        training=replace(base.training, layer_counts=(2,), interference_probability=0.0),
        model=replace(
            base.model,
            detector_channels=8,
            detector_sections=1,
            demapper_widths=(4, 4, 4, 4),
            denoise_widths=(8, 8, 8, 2),
            denoise_subsamples=(1, 2, 2, 1),
        ),
    )


def test_audit_payload_reports_valid_checkpoint_and_rejects_counter_mismatch(tmp_path):
    config = _tiny_config()
    path = tmp_path / "valid.pt"
    train_steps(
        EqDeepRx(config),
        OFDMSystem(config),
        config,
        steps=1,
        batch_size=1,
        n_layers=2,
        pilot_count=1,
        snr_db=12.0,
        seed=71,
        output_path=path,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    signature = {
        "batch_size": 1,
        "generation_batch_size": 1,
        "microbatch_size": 1,
        "n_layers": 2,
        "pilot_count": 1,
        "snr_db": 12.0,
        "seed": 71,
    }
    report = audit_payload(payload, config=config, run_signature=signature, amp_enabled=False)
    assert report["valid"] is True
    assert report["next_step"] == 1

    payload["optimizer_state_dict"]["state"][0]["step"] = 0
    broken = audit_payload(payload, config=config, run_signature=signature, amp_enabled=False)
    assert broken["valid"] is False
    assert any("optimizer state consistency" in reason for reason in broken["reasons"])


def test_audit_payload_rejects_optimizer_hyperparameter_mismatch(tmp_path):
    config = _tiny_config()
    path = tmp_path / "valid-hyperparams.pt"
    train_steps(
        EqDeepRx(config),
        OFDMSystem(config),
        config,
        steps=1,
        batch_size=1,
        n_layers=2,
        pilot_count=1,
        snr_db=12.0,
        seed=72,
        output_path=path,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    signature = {
        "batch_size": 1,
        "generation_batch_size": 1,
        "microbatch_size": 1,
        "n_layers": 2,
        "pilot_count": 1,
        "snr_db": 12.0,
        "seed": 72,
    }
    payload["optimizer_state_dict"]["param_groups"][0]["betas"] = (0.8, 0.999)
    report = audit_payload(
        payload, config=config, run_signature=signature, amp_enabled=False
    )
    assert report["valid"] is False
    assert any("optimizer hyperparameters" in reason for reason in report["reasons"])


def test_audit_payload_rejects_learning_rate_schedule_mismatch(tmp_path):
    config = _tiny_config()
    path = tmp_path / "valid-lr.pt"
    train_steps(
        EqDeepRx(config),
        OFDMSystem(config),
        config,
        steps=1,
        batch_size=1,
        n_layers=2,
        pilot_count=1,
        snr_db=12.0,
        seed=73,
        output_path=path,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    signature = {
        "batch_size": 1,
        "generation_batch_size": 1,
        "microbatch_size": 1,
        "n_layers": 2,
        "pilot_count": 1,
        "snr_db": 12.0,
        "seed": 73,
    }
    payload["optimizer_state_dict"]["param_groups"][0]["lr"] = 0.123
    report = audit_payload(
        payload, config=config, run_signature=signature, amp_enabled=False
    )
    assert report["valid"] is False
    assert any("learning-rate schedule" in reason for reason in report["reasons"])
