from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple


SUPPORTED_MODULATIONS = ("16QAM", "64QAM", "256QAM")
FIGURE6A_SINR_POINTS = tuple(float(value) for value in range(-5, 14, 2))


@dataclass(frozen=True)
class ChannelConfig:
    """Executable choices that the paper leaves to the implementation."""

    uma_delay_spread_mode: str = "native"
    # Table II exposes a 10--1100 ns range in the validation column. Keep the
    # main CDL validation path on that sampled range; fixed values are
    # exploratory. The training channel remains native UMa by default.
    cdl_delay_spread_mode: str = "uniform_10_1100ns"
    cir_discretization: str = "sinc"
    cir_normalization: bool = True
    enable_pathloss: bool = False
    enable_shadow_fading: bool = False
    interferer_timing: str = "random_symbol"


@dataclass(frozen=True)
class ModelConfig:
    """EqDeepRx Table I model widths and subsampling settings."""

    denoise_widths: Tuple[int, ...] = (64, 64, 64, 2)
    denoise_subsamples: Tuple[int, ...] = (1, 4, 2, 1)
    # Section III leaves the mixed-channel subset C_s qualitative; keep the
    # selected paper-aligned baseline explicit.
    time_mixer_channels: int = 2
    detector_channels: int = 64
    detector_sections: int = 4
    detector_subsample: int = 8
    demapper_widths: Tuple[int, ...] = (32, 32, 32, 8)
    max_bits: int = 8
    # The paper specifies a 1x1 skip projection when channel counts differ,
    # but does not publish whether that projection has a bias term.
    residual_projection_bias: bool = False


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
    # The paper requires coverage of all three layer counts but does not
    # publish their sampling probabilities; use the uniform baseline.
    layer_sampling_probabilities: Tuple[float, ...] = (
        1.0 / 3.0,
        1.0 / 3.0,
        1.0 / 3.0,
    )
    # Probability of selecting the two-DMRS configuration when the training
    # step does not pin a pilot count. The paper publishes both configurations,
    # not their mixture ratio.
    pilot_sampling_probability_two: float = 0.5
    interference_probability: float = 0.5
    # Table II publishes the lognormal INR location and spread. Keep them
    # explicit so both backends and the fidelity manifest use one source.
    inr_mean_db: float = 10.0
    inr_std_db: float = 5.0
    vcl_alpha: float = 1e-5
    seed: int = 2026
    backend: str = "sionna_tr38901"
    amp_dtype: str = "bfloat16"
    lamb_bias_correction: bool = True
    vcl_attachment: str = "all"
    symbol_loss_reduction: str = "sum"


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
    sinr_db_points: Tuple[float, ...] = FIGURE6A_SINR_POINTS
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
    # Table II counts one physical TX antenna per UE; n_tx_antennas is the
    # maximum number of one-antenna layers represented in the shared model.
    ue_tx_antennas: int = 1
    carrier_frequency_hz: float = 3.5e9
    maximum_channel_delay_s: float = 14e-6
    training_channel: str = "UMa"
    validation_channels: Tuple[str, ...] = ("UMa", "CDL-C", "CDL-D", "UMi")
    modulation: str = "64QAM"
    snr_db_range: Tuple[float, float] = (0.0, 45.0)
    speed_mps_range: Tuple[float, float] = (0.0, 35.0)
    delay_spread_ns_range: Tuple[float, float] = (10.0, 1100.0)
    # CDL profiles require a separate RMS delay-spread choice. The paper's
    # Table II validation entry is 10--1100 ns; the exact CDL sampling
    # plumbing is not otherwise specified.
    cdl_delay_spread_ns: float = 300.0
    interference_probability: float = 0.5
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    receiver: ReceiverConfig = ReceiverConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    channel: ChannelConfig = ChannelConfig()

    def __post_init__(self) -> None:
        """Keep legacy top-level training aliases synchronized.

        Earlier revisions exposed layer/interference sampling both at the
        system and training levels.  Accept either spelling for compatibility,
        but reject conflicting explicit values instead of silently using one.
        """

        default_layers = TrainingConfig().layer_counts
        default_layer_probabilities = TrainingConfig().layer_sampling_probabilities
        top_layers_changed = self.layer_counts != default_layers
        training_layers_changed = self.training.layer_counts != default_layers
        if top_layers_changed and training_layers_changed and self.layer_counts != self.training.layer_counts:
            raise ValueError("layer_counts and training.layer_counts disagree")
        if top_layers_changed and not training_layers_changed:
            probabilities = self.training.layer_sampling_probabilities
            if probabilities == default_layer_probabilities:
                probabilities = tuple(1.0 / len(self.layer_counts) for _ in self.layer_counts)
            object.__setattr__(
                self,
                "training",
                replace(
                    self.training,
                    layer_counts=self.layer_counts,
                    layer_sampling_probabilities=probabilities,
                ),
            )
        elif training_layers_changed and not top_layers_changed:
            object.__setattr__(self, "layer_counts", self.training.layer_counts)
            if self.training.layer_sampling_probabilities == default_layer_probabilities:
                object.__setattr__(
                    self,
                    "training",
                    replace(
                        self.training,
                        layer_sampling_probabilities=tuple(
                            1.0 / len(self.training.layer_counts)
                            for _ in self.training.layer_counts
                        ),
                    ),
                )
        elif top_layers_changed and training_layers_changed:
            if self.training.layer_sampling_probabilities == default_layer_probabilities:
                object.__setattr__(
                    self,
                    "training",
                    replace(
                        self.training,
                        layer_sampling_probabilities=tuple(
                            1.0 / len(self.training.layer_counts)
                            for _ in self.training.layer_counts
                        ),
                    ),
                )

        default_probability = TrainingConfig().interference_probability
        top_probability_changed = self.interference_probability != default_probability
        training_probability_changed = self.training.interference_probability != default_probability
        if top_probability_changed and training_probability_changed and self.interference_probability != self.training.interference_probability:
            raise ValueError("interference_probability and training.interference_probability disagree")
        if top_probability_changed and not training_probability_changed:
            object.__setattr__(self, "training", replace(self.training, interference_probability=self.interference_probability))
        elif training_probability_changed and not top_probability_changed:
            object.__setattr__(self, "interference_probability", self.training.interference_probability)

        if self.channel.uma_delay_spread_mode not in {"native", "uniform_10_1100ns"}:
            raise ValueError("unsupported UMa delay-spread mode")
        if self.channel.cdl_delay_spread_mode not in {
            "uniform_10_1100ns",
            "fixed",
        }:
            raise ValueError("unsupported CDL delay-spread mode")
        if self.channel.cir_discretization not in {"sinc", "nearest"}:
            raise ValueError("unsupported CIR discretization")
        if self.channel.interferer_timing not in {"random_symbol", "zero", "random_sample"}:
            raise ValueError("unsupported interferer timing mode")
        if self.training.amp_dtype not in {"bfloat16", "float32"}:
            raise ValueError("amp_dtype must be bfloat16 or float32")
        if self.training.vcl_attachment not in {"all", "final", "none"}:
            raise ValueError("vcl_attachment must be all, final, or none")
        if not self.training.layer_counts:
            raise ValueError("layer_counts must not be empty")
        if len(self.training.layer_sampling_probabilities) != len(self.training.layer_counts):
            raise ValueError(
                "layer_sampling_probabilities must match layer_counts length"
            )
        if any(
            not 0.0 < float(value) <= 1.0
            for value in self.training.layer_sampling_probabilities
        ):
            raise ValueError("layer sampling probabilities must be finite and positive")
        if abs(sum(float(value) for value in self.training.layer_sampling_probabilities) - 1.0) > 1e-6:
            raise ValueError("layer sampling probabilities must sum to one")
        if not 0.0 <= self.training.pilot_sampling_probability_two <= 1.0:
            raise ValueError("pilot_sampling_probability_two must be between 0 and 1")
        if self.training.symbol_loss_reduction not in {"sum", "mean_active"}:
            raise ValueError("unsupported symbol-loss reduction")
        if self.model.time_mixer_channels < 1:
            raise ValueError("time_mixer_channels must be positive")
        if self.training.inr_std_db < 0:
            raise ValueError("inr_std_db must be nonnegative")
        if not 0.0 <= self.training.interference_probability <= 1.0:
            raise ValueError("interference_probability must be between 0 and 1")
        if self.cdl_delay_spread_ns <= 0:
            raise ValueError("cdl_delay_spread_ns must be positive")
        if self.ue_tx_antennas != 1:
            raise ValueError("the paper protocol uses one TX antenna per UE")

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
