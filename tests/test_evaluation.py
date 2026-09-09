from dataclasses import replace

from eqdeeprx.config import paper_config
from eqdeeprx.evaluation import evaluate_uncoded_ber
from eqdeeprx.model import EqDeepRx
from eqdeeprx.signal import OFDMSystem


def test_uncoded_ber_evaluation_returns_five_finite_curves(tmp_path):
    base = paper_config().with_modulation("16QAM")
    config = replace(
        base,
        n_subcarriers=16,
        n_fft=24,
        cyclic_prefix=4,
        n_rx_antennas=2,
        n_tx_antennas=2,
        layer_counts=(2,),
        model=replace(base.model, detector_channels=8, detector_sections=1, demapper_widths=(4, 4, 4, 4), denoise_widths=(8, 8, 8, 2), denoise_subsamples=(1, 2, 2, 1)),
    )
    metrics = evaluate_uncoded_ber(EqDeepRx(config), OFDMSystem(config), config, snr_points=(0.0, 6.0), samples_per_point=1, n_layers=2, seed=4, output_dir=tmp_path)

    assert set(metrics["curves"]) == {"eqdeeprx_1_pilot", "eqdeeprx_2_pilots", "lmmse_1_pilot", "lmmse_2_pilots", "lmmse_known_channel"}
    assert all(len(values) == 2 for values in metrics["curves"].values())
    assert all(value == value and value >= 0.0 for values in metrics["curves"].values() for value in values)
    assert (tmp_path / "uncoded_ber_metrics.json").exists()

