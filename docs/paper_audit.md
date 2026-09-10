# Paper Alignment Audit

The final implementation was checked against the v2 LaTeX source of *EqDeepRx: Learning a Scalable and Interference Mitigating MIMO Receiver* and the supplied DeepRx paper/code. Scope is the primary EqDeepRx 64-QAM path through uncoded BER; decoder-dependent and ablation experiments are excluded.

## Published Equations and Flow

| Paper requirement | Implementation | Result |
|---|---|---|
| Eq. (2)-(3): RZF `(H^H H + alpha I)^-1 H^H`, `alpha=1e-4`, inverse diagonal unit-gain scaling | `receiver.py:rzf_equalize` | Matched |
| Eq. (5)-(7): `R=E[dd^H]`, LMMSE with `R`, inverse diagonal unit-gain scaling | `receiver.py:estimate_incm`, `lmmse_equalize` | Matched |
| Eq. (8): rank-one `y x^H / ||x||^2` at orthogonal DMRS REs | `receiver.py:estimate_raw_channel` | Matched |
| Eq. (9): SCM per 24 subcarriers and complex-Gaussian OAS shrinkage toward `tr(S)/N_R I` | `receiver.py:oas_complex_shrinkage`, `estimate_incm` | Matched |
| Eq. (10): LLR `log P(bit=0)/P(bit=1)` | demappers, BCE, BER decisions | Matched; positive means bit zero |
| Eqs. (11)-(12): nearest 1:N down/up sampling, full-resolution residual, asymmetric depthwise-separable convolutions | `layers.py:SubsampledResidualBlock` | Matched |
| Training loss: SNR weight, BCE, squared symbol norm from every Detector section, `lambda=1e-5` | `losses.py:eqdeeprx_loss` | Matched |
| mVCL: per-channel batch/spatial mean 0 and variance 1, divide by channel count, `alpha=1e-5` | `losses.py`, exact accumulated-batch statistics in `training.py` | Matched |

The standard data path is full time-domain Sionna 2.1: online UMa training; independent desired/interfering TR 38.901 channels; OFDM, CP, random interferer timing, AWGN, and resulting ISI/ICI; receiver processing and uncoded hard-bit BER then operate on the demodulated full slot.

## Published Architecture and Parameters

- DenoiseNN: independent RX/TX pilot pair, real/imag inputs, four frequency-only residual blocks, widths `[64,64,64,2]`, subsampling `[1,4,2,1]`, and pointwise time mixing after each block.
- DetectorNN: six real inputs (LMMSE, RZF, frequency map, time map), `1x1` projection to 64, four shared-weight sections, each with full and 1:8 residual blocks.
- DemapperNN: four shared per-layer pointwise residual blocks, widths `[32,32,32,8]`; lower modulation orders mask unused outputs.
- System: 30 kHz SCS, 192 subcarriers, 14 symbols, 16 RX antennas, 2-4 one-antenna UE layers, UMa training, and UMa/CDL-C/CDL-D/UMi validation.
- Distribution: SNR uniform 0-45 dB, speed uniform 0-35 m/s, lognormal INR `(10 dB, 5 dB)`, zero/one interfering UE, orthogonal staggered DMRS on every fourth subcarrier, one/two DMRS symbols.
- Training: effective batch 112, LAMB, initial LR `4.4e-3`, linear decay to zero, about 70k iterations/about 8M samples, 64-QAM primary model.
- Scale invariance: one training job samples 2/3/4 layers and both DMRS patterns; the same DetectorNN/DemapperNN modules are called per layer with shared parameters. No layer-specific network or checkpoint is created.
- Primary learned parameter count: **115,456**, which rounds to the paper's 116k value (equalizers excluded).

## DeepRx Reuse

The supplied DeepRx work established the full-slot tensor convention, DMRS-to-channel-estimate boundary, direct bit-supervised training, and uncoded-BER endpoint. EqDeepRx is not the monolithic DeepRx CNN: it replaces that learned path with DenoiseNN, two conventional equalizers, shared per-layer DetectorNN, and DemapperNN as required by the newer paper. The supplied DeepRx tree was left unchanged.

## Figure 6(a)

`evaluation.py:evaluate_paper_figure6a` requires the Sionna backend and a trained checkpoint, uses CDL-C at 10-15 m/s with one interferer, samples requested SNR and bins errors by realized SINR, and uses 32,000 validation slots total across the two DMRS cases. It produces uncoded BER before LDPC; DenoiseNN-only and decoder-dependent curves are not implemented.

## Necessary Reproduction Choices

The public paper does not specify the original FFT/CP values, TimeMixer subset `C_s`, LAMB beta/epsilon/weight decay, random seed/frame order, exact SINR bin edges, or which internal layers receive mVCL. This implementation fixes these at 256/18, 2, `(0.9,0.999)/1e-6/0`, 2026, one-dB bins, and all four Detector section states. These choices are configuration-tested and preflight-tested, but cannot be asserted to match unavailable author-private code bit for bit.
