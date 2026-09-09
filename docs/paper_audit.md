# EqDeepRx Paper and DeepRx Audit

## Paper facts used by the implementation

The local paper PDF was read page by page. The implementation follows these equations and architecture statements:

| Paper item | Code location | Audit result |
|---|---|---|
| Raw rank-one DMRS estimate, Eq. (8) | `src/eqdeeprx/receiver.py:estimate_raw_channel` | `y x* / ||x||^2` at pilot REs |
| RZF, Eq. (2)-(3) | `receiver.py:rzf_equalize` | `alpha=1e-4`, unit-gain diagonal scaling |
| INCM and shrinkage, Eq. (5),(9) | `receiver.py:estimate_incm` | 24-subcarrier bands, Hermitian PSD shrinkage |
| LMMSE, Eq. (6)-(7) | `receiver.py:lmmse_equalize` | Uses the estimated INCM and unit-gain scaling |
| DenoiseNN | `model.py:DenoiseNN` | Per RX/TX pair, frequency-only separable residual blocks, `[64,64,64,2]`, subsampling `[1,4,2,1]` |
| DetectorNN | `model.py:DetectorNN` | Four sections, shared across layers, 6 real input channels (two equalizers plus two coordinate maps), 1:8 residual path |
| DemapperNN | `model.py:DemapperNN` | Four 1x1 residual blocks, `[32,32,32,8]`, shared across layers |
| Training loss, Eq. (13) | `losses.py:eqdeeprx_loss` | Positive-logit bit-one BCE, `log2(1+SNR)` weighting, `lambda=1e-5` symbol loss |
| VCL stability regularization | `losses.py:vcl_regularization` and `training.py:train_steps` | Per-channel batch/spatial mean target 0 and variance target 1; `alpha=1e-5` mean term, applied to every DetectorNN section state |
| Paper uncoded BER endpoint | `evaluation.py` and `scripts/evaluate_uncoded_ber.py` | Stops before decoder, as requested |

The default system is the paper's 30 kHz/192-subcarrier/16-RX/2-4-layer setup, not the supplied DeepRx reproduction's 15 kHz/312-subcarrier/2-RX/1-layer setup. A tiny configuration exists only for tests and smoke runs.

The paper-scale training loop samples 2, 3, or 4 MIMO layers, one or two DMRS symbols, and an interference-present batch with probability 0.5. Explicit `n_layers`, `pilot_count`, and SNR arguments remain available for deterministic smoke tests.

The full default model count measured locally is **118,740** trainable parameters, excluding the deterministic equalizers.

## Inheritance relationship with DeepRx

EqDeepRx is an extension of the earlier DeepRx idea, not a drop-in replacement for the DeepRx CNN:

1. Both start from a full OFDM resource grid and DMRS-derived channel information, retain conventional communication-system operations, and train directly against transmitted bits using positive-logit bit convention.
2. DeepRx's original path feeds received signal, pilots, and raw LS features into one monolithic fully convolutional ResNet that emits all LLRs together.
3. EqDeepRx keeps raw DMRS estimation/interpolation but inserts learned pilot-domain denoising, then runs two conventional equalizers (RZF and INCM-aware LMMSE) in parallel. The learned detector and demapper are applied with shared weights independently to each MIMO layer.
4. This decomposition is the source of EqDeepRx's near-linear layer scaling and layer-count generalization. The implementation therefore preserves a layer dimension in logits for multi-layer runs; a one-layer call is squeezed to the familiar `[N,B,F,S]` DeepRx shape.

The supplied DeepRx repository remains unchanged. Its MATLAB path is standards-grade and its Python reference confirms the tensor layout and uncoded BER boundary, but its published 30k-step checkpoint cannot be used as an EqDeepRx checkpoint because the architectures and inputs differ.

## Scope and limitations

- This delivery implements the uncoded BER stage only. LDPC encoding/decoding, BLER, MCS-table rate matching, and spectral-efficiency curves are deliberately outside the requested handoff.
- The local Python link uses an OFDM-equivalent CP-sufficient multipath channel for fast deterministic smoke/evaluation runs. A standards-grade MATLAB/Sionna channel can be connected at the `SignalBatch` boundary without changing the model or receiver APIs.
- The paper's original online 8M-sample dataset seed and exact frame order are not public; this project fixes seed 2026 and records every smoke/evaluation seed.
