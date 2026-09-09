# EqDeepRx Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained PyTorch reproduction of the EqDeepRx receiver up to uncoded BER, using the existing DeepRx OFDM link as the reference integration style while freezing the EqDeepRx paper's architecture and training defaults.

**Architecture:** The new `eqdeeprx` project separates deterministic paper configuration, QAM/OFDM/channel generation, conventional DMRS channel estimation and INCM-aware RZF/LMMSE equalization, and the three learned stages (DenoiseNN, shared per-layer DetectorNN, and DemapperNN). The training path uses direct uncoded QAM bits, paper-weighted BCE plus symbol loss, deterministic online batches, and explicit smoke/preflight commands; LDPC decoding is intentionally outside scope.

**Tech Stack:** Python 3.9+, PyTorch 2.8+, NumPy, SciPy-free torch signal processing, pytest, Matplotlib, setuptools.

## Global Constraints

- EqDeepRx primary architecture: DenoiseNN per RX/TX pair, parallel LMMSE and RZF equalizers, shared per-MIMO-layer DetectorNN, and per-layer DemapperNN.
- Paper signal defaults: OFDM 30 kHz SCS, 192 active subcarriers, 14 symbols/slot, 16 RX antennas, 2-4 layers, UMa training, 64-QAM primary runs and a configurable 16-QAM uncoded-BER run.
- Paper equalizer defaults: RZF regularization `alpha=1e-4`, 24-subcarrier interference coherence bandwidth, complex covariance shrinkage, unit-gain post-scaling.
- Paper model defaults: DenoiseNN filters `[64,64,64,2]`, depth 4, frequency-only `N x 1` kernels; DetectorNN 4 sections with 64 channels and full/1:8 residual paths; DemapperNN `[32,32,32,8]` 1x1 residual blocks.
- Paper training defaults: batch size 112, LAMB, learning rate `4.4e-3`, linear decay to zero, approximately 70,000 iterations, symbol-loss weight `lambda=1e-5`; smoke tests must override these to tiny values.
- Outputs stop at uncoded BER. No LDPC encoder/decoder is required in the new project.
- Large cache generation and full training are never started automatically.
- Git excludes checkpoints, generated datasets, caches, plots, and local environments.

---

### Task 1: Project scaffold and paper configuration

**Files:**
- Create: `eqdeeprx/pyproject.toml`
- Create: `eqdeeprx/requirements.txt`
- Create: `eqdeeprx/.gitignore`
- Create: `eqdeeprx/src/eqdeeprx/__init__.py`
- Create: `eqdeeprx/src/eqdeeprx/config.py`
- Test: `eqdeeprx/tests/test_config.py`

**Interfaces:**
- Produces `EqDeepRxConfig`, `ModelConfig`, `TrainingConfig`, and `ReceiverConfig` dataclasses plus `paper_config()`.

- [x] **Step 1: Write the failing test** asserting exact paper defaults, validation of unsupported modulation/layer counts, and no accidental 312-subcarrier DeepRx defaults.
- [x] **Step 2: Run `py -3 -m pytest tests/test_config.py -q` and observe the missing-module failure.
- [x] **Step 3: Add typed frozen dataclasses and validation with the exact constants listed above.
- [x] **Step 4: Re-run the focused test and then `py -3 -m pytest tests/test_config.py -q`; expect all configuration tests to pass.
- [x] **Step 5: Commit `feat: add EqDeepRx reproduction up to uncoded BER`** (combined implementation commit `08bc7ba`).

### Task 2: Signal generation and conventional receiver math

**Files:**
- Create: `eqdeeprx/src/eqdeeprx/constellation.py`
- Create: `eqdeeprx/src/eqdeeprx/signal.py`
- Create: `eqdeeprx/src/eqdeeprx/receiver.py`
- Test: `eqdeeprx/tests/test_signal.py`
- Test: `eqdeeprx/tests/test_receiver.py`

**Interfaces:**
- `qam_modulate(bits, modulation) -> complex symbols` and `qam_demapper_llr(symbols, noise_var, modulation, max_bits) -> [B,...]`.
- `OFDMSystem.transmit(bits, seed, pilot_count, n_layers) -> Batch` with tensors `[batch, n_rx, F, S]`, `[batch, n_layers, F, S]`, pilot grid, target bits, data mask, and channel truth.
- `estimate_raw_channel`, `interpolate_channel`, `estimate_incm`, `rzf_equalize`, `lmmse_equalize`, and `equalize_parallel` implement equations (2)-(9) and unit-gain scaling.

- [x] **Step 1: Write failing tests for Gray QAM round trips, OFDM shape/mask invariants, raw LS estimates, identity-channel RZF/LMMSE, Hermitian PSD covariance, and finite increasing-SNR BER.
- [x] **Step 2: Run both focused files and confirm the expected missing-module failures.
- [x] **Step 3: Implement deterministic QAM, pilot placement (orthogonal staggered layer pilots), OFDM IFFT/FFT with cyclic prefix, a compact TDL/UMa-like fading channel with optional interference, and the receiver equations.
- [x] **Step 4: Run focused tests; fix numerical issues until all pass on CPU.
- [x] **Step 5: Included in combined implementation commit `08bc7ba`.

### Task 3: EqDeepRx learned modules

**Files:**
- Create: `eqdeeprx/src/eqdeeprx/layers.py`
- Create: `eqdeeprx/src/eqdeeprx/model.py`
- Test: `eqdeeprx/tests/test_model.py`

**Interfaces:**
- `SubsampledResidualBlock(in_channels, out_channels, downsample, frequency_only)`.
- `DenoiseNN(in_channels=2, widths=(64,64,64,2), subsamples=(1,4,2,1))` returns denoised pilot estimates with the same pilot-grid shape.
- `DetectorNN(channels=64, sections=4, bits=8)` returns final features and one symbol estimate per section.
- `DemapperNN(in_channels=64, widths=(32,32,32,8))` returns per-layer logits.
- `EqDeepRx.forward(received_grid, pilot_grid, pilot_mask, n_layers=None, return_aux=False)` returns `[N,B,F,S]` logits and optional symbol states/equalized streams.

- [x] **Step 1: Write failing tests for module output shapes, shared-weight layer invariance, variable frequency width, finite gradients, and default parameter count within the paper's approximately 116k range (equalizers excluded).
- [x] **Step 2: Run `py -3 -m pytest tests/test_model.py -q` and confirm failure before implementation.
- [x] **Step 3: Implement preactivation/separable residual blocks, nearest-neighbor 1:8 residual subsampling, pilot-domain time mixing, coordinate maps, shared per-layer processing, and 8-bit demapper output.
- [x] **Step 4: Run focused model tests and verify no NaN/shape regressions.
- [x] **Step 5: Included in combined implementation commit `08bc7ba`.

### Task 4: Paper loss, deterministic training, and evaluation

**Files:**
- Create: `eqdeeprx/src/eqdeeprx/losses.py`
- Create: `eqdeeprx/src/eqdeeprx/training.py`
- Create: `eqdeeprx/src/eqdeeprx/evaluation.py`
- Create: `eqdeeprx/scripts/train.py`
- Create: `eqdeeprx/scripts/evaluate_uncoded_ber.py`
- Create: `eqdeeprx/scripts/preflight.py`
- Test: `eqdeeprx/tests/test_training.py`
- Test: `eqdeeprx/tests/test_evaluation.py`

**Interfaces:**
- `eqdeeprx_loss(logits, target_bits, data_mask, bit_mask, symbol_states, target_symbols, snr_linear, lambda_symbol)` implements equation (13) with positive-logit bit-1 convention.
- `Lamb` and `paper_learning_rate(step, total_steps, base_lr)` expose the paper schedule.
- `train_steps(model, batches, config, device)` performs one or more deterministic steps and writes atomic checkpoints.
- `evaluate_uncoded_ber(model, snr_points, samples_per_point, seed, ...)` returns JSON-serializable five-curve metrics (EqDeepRx 1/2 pilot, LMMSE 1/2 pilot, known-channel LMMSE).
- `preflight.py` validates dependency, model, signal-chain, memory, and device readiness without training at paper scale.

- [x] **Step 1: Write failing tests for positive-logit BCE, symbol loss weighting, LAMB hyperparameters, warmup/linear-decay endpoints, deterministic batches/checkpoints, resume metadata, and finite BER JSON.
- [x] **Step 2: Run focused tests and confirm missing implementation failures.
- [x] **Step 3: Implement the weighted loss, optional VCL channel-stat regularization, LAMB, deterministic seeded batch generation, atomic checkpointing, short evaluation, and preflight resource estimate.
- [x] **Step 4: Run focused tests, then a one-step CPU smoke command with tiny dimensions.
- [x] **Step 5: Included in combined implementation commit `08bc7ba`.

### Task 5: Documentation and audits

**Files:**
- Create: `eqdeeprx/README.md`
- Create: `eqdeeprx/docs/paper_audit.md`
- Create: `eqdeeprx/docs/open_source_search.md`
- Create: `eqdeeprx/docs/preflight_report.md`
- Create: `eqdeeprx/tests/test_documentation.py`

- [x] **Step 1: Write failing documentation tests for required commands, explicit decoder scope, parameter table, and search evidence.
- [x] **Step 2: Run the focused documentation test and confirm missing-file failure.
- [x] **Step 3: Document the paper-to-code mapping, DeepRx inheritance/differences, failed/limited external search evidence, exact smoke commands, expected GPU memory, and the command that starts (but is not run by default) full training.
- [x] **Step 4: Run documentation tests and manually inspect the rendered Markdown text for contradictions.
- [x] **Step 5: Included in combined implementation commit `08bc7ba`.

### Task 6: Full verification and GitHub handoff

**Files:**
- Modify: `eqdeeprx/.gitignore`
- Create: `eqdeeprx/outputs/.gitkeep`

- [x] **Step 1: Run the complete pytest suite with `py -3 -m pytest -q`.
- [x] **Step 2: Run `py -3 scripts/preflight.py --tiny` and `py -3 scripts/train.py --steps 1 --batch-size 2 --tiny`.
- [x] **Step 3: Run `py -3 scripts/evaluate_uncoded_ber.py --snr-points 0,6 --samples-per-point 2 --tiny` and validate finite JSON/PNG outputs.
- [x] **Step 4: Re-read the paper audit and requirement checklist; verify no decoder/full-scale training was run and generated artifacts are ignored.
- [x] **Step 5a: Initialize `eqdeeprx` as its own Git repository, add the requested `origin`, and create clean commit `08bc7ba` after local tests passed.
- [ ] **Step 5b: Push to `https://github.com/Tonyerwite/EqDeepRX.git`; currently blocked because terminal HTTPS is reset/times out and the GitHub browser page is signed out (SSH has no public key). The local repository is ready for an authenticated `git push -u origin main`.

## Self-Review Checklist

- Paper constants appear in `config.py`, tests, and `docs/paper_audit.md`.
- Every new production module has a test written and observed failing before implementation.
- The model uses both LMMSE and RZF, denoises raw DMRS estimates, processes each layer with shared weights, and exposes 8-bit logits.
- Evaluation reports uncoded BER only and does not claim LDPC/BLER reproduction.
- Full 70k-step training is documented but never started automatically.
- Git excludes caches/checkpoints and contains only source, tests, docs, and small smoke outputs.
