from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple


SUPPORTED_MODULATIONS = ("16QAM", "64QAM", "256QAM")


@dataclass(frozen=True)
class ModelConfig:
    """EqDeepRx Table I model widths and subsampling settings."""

    denoise_widths: Tuple[int, ...] = (64, 64, 64, 2)
    denoise_subsamples: Tuple[int, ...] = (1, 4, 2, 1)
    detector_channels: int = 64
    detector_sections: int = 4
    detector_subsample: int = 8
    demapper_widths: Tuple[int, ...] = (32, 32, 32, 8)
    max_bits: int = 8


@dataclass(frozen=True)
class TrainingConfig:
    """Paper Section IV training hyperparameters."""

    batch_size: int = 112
    learning_rate: float = 4.4e-3
    total_steps: int = 70_000
    warmup_steps: int = 0
    weight_decay: float = 0.0
    lamb_beta1: float = 0.9
    lamb_beta2: float = 0.999
    lamb_eps: float = 1e-6
    symbol_loss_weight: float = 1e-5
    layer_counts: Tuple[int, ...] = (2, 3, 4)
    interference_probability: float = 0.5
    vcl_alpha: float = 1e-5
    seed: int = 2026
    backend: str = "sionna_tr38901"


@dataclass(frozen=True)
class ReceiverConfig:
    """Deterministic receiver operations from EqDeepRx Sections II-III."""

    rzf_alpha: float = 1e-4
    incm_coherence_bandwidth: int = 24
    shrinkage: str = "oas"
    baseline_smoothing_window: int = 5
    pilot_spacing: int = 4
    dmrs_symbols: Tuple[int, ...] = (2, 11)


@dataclass(frozen=True)
class EvaluationConfig:
    """Protocol of the paper's interference-present Fig. 6(a)."""

    channel_model: str = "CDL-C"
    speed_mps_range: Tuple[float, float] = (10.0, 15.0)
    sinr_db_points: Tuple[float, ...] = tuple(float(value) for value in range(-5, 11))
    sinr_bin_width_db: float = 1.0
    validation_samples: int = 32_000
    interfering_ues: int = 1
    pilot_counts: Tuple[int, ...] = (1, 2)


@dataclass(frozen=True)
class EqDeepRxConfig:
    """Paper-aligned system defaults for the uncoded-BER reproduction."""

    subcarrier_spacing_khz: int = 30
    n_subcarriers: int = 192
    n_fft: int = 256
    cyclic_prefix: int = 18
    n_ofdm_symbols: int = 14
    n_rx_antennas: int = 16
    layer_counts: Tuple[int, ...] = (2, 3, 4)
    n_tx_antennas: int = 4
    carrier_frequency_hz: float = 3.5e9
    maximum_channel_delay_s: float = 14e-6
    training_channel: str = "UMa"
    validation_channels: Tuple[str, ...] = ("UMa", "CDL-C", "CDL-D", "UMi")
    modulation: str = "64QAM"
    snr_db_range: Tuple[float, float] = (0.0, 45.0)
    speed_mps_range: Tuple[float, float] = (0.0, 35.0)
    delay_spread_ns_range: Tuple[float, float] = (10.0, 1100.0)
    interference_probability: float = 0.5
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    receiver: ReceiverConfig = ReceiverConfig()
    evaluation: EvaluationConfig = EvaluationConfig()

    def __post_init__(self) -> None:
        """Keep legacy top-level training aliases synchronized.

        Earlier revisions exposed layer/interference sampling both at the
        system and training levels.  Accept either spelling for compatibility,
        but reject conflicting explicit values instead of silently using one.
        """

        default_layers = TrainingConfig().layer_counts
        top_layers_changed = self.layer_counts != default_layers
        training_layers_changed = self.training.layer_counts != default_layers
        if top_layers_changed and training_layers_changed and self.layer_counts != self.training.layer_counts:
            raise ValueError("layer_counts and training.layer_counts disagree")
        if top_layers_changed and not training_layers_changed:
            object.__setattr__(self, "training", replace(self.training, layer_counts=self.layer_counts))
        elif training_layers_changed and not top_layers_changed:
            object.__setattr__(self, "layer_counts", self.training.layer_counts)

        default_probability = TrainingConfig().interference_probability
        top_probability_changed = self.interference_probability != default_probability
        training_probability_changed = self.training.interference_probability != default_probability
        if top_probability_changed and training_probability_changed and self.interference_probability != self.training.interference_probability:
            raise ValueError("interference_probability and training.interference_probability disagree")
        if top_probability_changed and not training_probability_changed:
            object.__setattr__(self, "training", replace(self.training, interference_probability=self.interference_probability))
        elif training_probability_changed and not top_probability_changed:
            object.__setattr__(self, "interference_probability", self.training.interference_probability)

    @property
    def rzf_alpha(self) -> float:
        return self.receiver.rzf_alpha

    @property
    def incm_coherence_bandwidth(self) -> int:
        return self.receiver.incm_coherence_bandwidth

    @property
    def sample_rate_hz(self) -> float:
        """OFDM sampling rate implied by FFT size and subcarrier spacing."""

        return float(self.n_fft * self.subcarrier_spacing_khz * 1000)

    def validate_layer_count(self, n_layers: int) -> None:
        if n_layers not in self.layer_counts:
            raise ValueError(f"layer count {n_layers} is outside paper-supported layers {self.layer_counts}")
        if n_layers > self.n_tx_antennas:
            raise ValueError("layer count cannot exceed configured transmit antennas")

    def with_modulation(self, modulation: str) -> "EqDeepRxConfig":
        normalized = modulation.upper()
        if normalized not in SUPPORTED_MODULATIONS:
            raise ValueError(f"Unsupported modulation: {modulation}; EqDeepRx supports {SUPPORTED_MODULATIONS}")
        return replace(self, modulation=normalized)


def paper_config() -> EqDeepRxConfig:
    """Return a fresh immutable configuration matching the paper defaults."""

    return EqDeepRxConfig()
