# [ARCHIVED] Clean, Resumable 70k Training and Verified Figure 6(a) Implementation Plan

> **Publication status:** This is the implementation history and next-run plan. The legacy checkpoint and legacy Figure 6(a) files named below are historical inputs that were rejected and are not repository release artifacts. The current repository intentionally publishes code, tests, audits, and executable plans only; runtime checkpoints, plots, and metrics remain on `D:` until a new clean run passes its gates.
>
> **Superseded publication instructions:** The artifact-publication wording in this archived plan is historical. The current user-approved release publishes source, tests, audits, and plans now, while keeping runtime checkpoints, plots, and metrics outside Git until a clean run is verified.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a defensible EqDeepRx `next_step=70000` checkpoint whose 70,000 entries are successful, state-consistent optimizer updates, then generate Figure 6(a) from that exact checkpoint and publish only verified artifacts.

**Architecture:** Keep the paper-aligned model, loss, data distribution, LAMB baseline, effective batch size, and 70,000-step target unchanged. Add a strict checkpoint contract at load/save boundaries and a finite-value boundary before and after every optimizer update. A read-only audit chooses the highest existing checkpoint that satisfies the contract; if none does, the run starts from a new initialized model. Separately, run a fixed-sample, one-variable sensitivity harness for the unpublished implementation choices; its outputs are exploratory and never replace the formal configuration. Checkpoints are atomic and resumable, so an interruption can discard at most the unsaved tail without inventing or skipping a committed global step.

**Tech Stack:** Python 3.12, PyTorch 2.9.1+cu128, CUDA AMP with the existing bfloat16 model boundaries, Sionna 2.1.0, pytest, PowerShell.

## Global Constraints

- Preserve the existing EqDeepRx architecture, loss equations, VCL regularization, LAMB settings, paper sampling distribution, effective batch size 112, generation batch size 2, approved microbatch size 28, seed 2026, and target of exactly 70,000 successful optimizer steps.
- Do not silently repair, truncate, pad, or reset an inconsistent checkpoint and then label it as a valid continuation.
- Do not overwrite the existing invalid checkpoint, anomaly backups, or prior evaluation outputs.
- A non-finite loss, gradient, optimizer state, or model tensor is a hard stop for the formal run. Preserve the anomaly and the last valid formal checkpoint; do not change hyperparameters to force progress.
- Do not start a long run or evaluation until the relevant verification gates pass. Do not rewrite README, commit, or push until the resulting artifacts pass independent verification.
- Every scheduled monitor report must state the actual checkpoint `next_step`, `history.steps`, latest loss, latest BER, process state, both log scans, GPU utilization/memory/temperature, C:/D: free space, and checkpoint write time.

## Confirmed Read-Only Audit

The following facts were established from the repository, the evaluator outputs, and every `.pt` file currently under `D:\EqDeepRxRuns\checkpoints`. They are evidence for the implementation plan, not a reason to publish the current result.

### Current `eqdeeprx.pt`

| Field | Observed value | Meaning |
| --- | --- | --- |
| `next_step` | `70000` | The file claims a 70,000 global step target. |
| top-level `steps` | `70000` | Same claim, but not sufficient by itself. |
| `history.steps` | `70000` | Same claim, but not sufficient by itself. |
| `history.losses`, `history.bers`, `history.learning_rates` | `69666` entries each | 334 committed history entries are missing. |
| optimizer states | 80 | All current parameters have some state, but their counters disagree. |
| denoise/detector state counters | 60 states at `69666` | These states reflect the post-recovery update count. |
| demapper state counters | 20 states at `55000` | These states were reset at the 15,000-step recovery and then advanced 55,000 times; this is not proof that Demapper stopped at global step 55,000. |
| GradScaler scale | `134217728` | Finite, but it does not repair the inconsistent history or optimizer counters. |
| model/optimizer tensors | finite | Finite final tensors do not make the trajectory a clean 70k run. |

The exact state-counter mapping is the 20 `demapper.*` parameters at 55,000 and the remaining 60 parameters at 69,666. The prior statement that "Demapper broke at 55,000" is therefore not established and will not be used as the root cause.

### Observed numerical failures and recovery artifacts

- `eqdeeprx_step15000_nan_backup.pt`: the first non-finite loss is at zero-based history index `14666`; the payload then has `next_step=15000`, a zero GradScaler scale, and an abnormally large Demapper norm (about `21657`). The old trainer advanced/recorded state after the bad update instead of rejecting it.
- `eqdeeprx_recovered_start.pt`: `next_step=15000`, each history array has `14666` entries, and only 60 optimizer states are present. The missing 20 Demapper states prove that the recovery process reinitialized that optimizer portion.
- `eqdeeprx_nan_step28000_20260914_010736.pt`: a later recovered run has its first non-finite loss at index `26136`, history length `27666` versus `next_step=28000`, and GradScaler scale `0`. This is independent evidence that the old finite-value boundary was not sufficient.
- `diagnostic_float32_step14700.pt`: all 80 states exist, but the 20 Demapper counters are `34` while the other 60 are `14700`; it is not a valid continuation despite finite tensors.
- The remaining recovery/test files have the same class of mismatch (truncated history, missing states, or unequal state counters). No current D: checkpoint satisfies the complete continuation contract described below.

The stored payloads do not contain a per-layer overflow trace, so the exact internal tensor that first overflowed cannot be proven retrospectively. The code and diagnostic experiments are consistent with float16 overflow in the high-dynamic-range detector/residual path; the existing `ecd8fd2` model boundary change to bfloat16/float32 reduces that risk for a new run. The checkpoint contract and pre/post-update finite guard are still required because the model boundary alone cannot repair a bad history or a partially reset optimizer.

### Why the 52,500 and 70,000 curves look similar

The comparison is reproducible and not an exact duplicate:

- Both runs use the same seed, protocol, realized-SINR samples, and bin counts. The baseline curves are byte-for-byte equal.
- The EqDeepRx curves differ. For example, the 1-pilot BER changes from `0.274014586129` to `0.271516726504` at `-5 dB`, and from `0.033200789495` to `0.032232072098` at `13 dB`. The 2-pilot values change from `0.255611986364` to `0.253221810508` and from `0.013774667748` to `0.013283372416`.
- The 70k evaluation progress fingerprint matches the model tensor fingerprint in the current `eqdeeprx.pt`; the 52.5k fingerprint does not. Thus the existing 70k image loaded the current 70k-labeled weights, while the weights themselves came from an inconsistent recovery trajectory.

The missing history entries do not directly alter an evaluator that loads only `model_state_dict`. The optimizer-state reset and the non-finite recovery events do alter subsequent training updates, so they are sufficient reason to reject the current file as a strict clean-70k result even though its plotted image is genuinely generated from its stored tensors.

The exact quantitative contribution of the reset to the visual plateau is not identifiable from the surviving files: there is no clean 52,500 checkpoint and no controlled counterfactual run with only the Demapper state changed. The plan therefore treats the plateau as an observation to re-test, not as a proven single-cause explanation.

### D: data inventory

The D: volume contains the EqDeepRx runtime/cache directories, the checkpoint candidates above, and a separate `external\DeepRX` reference tree with MATLAB/source and reference figure assets. It does not contain an EqDeepRx static validation dump or an additional valid 55k/70k checkpoint. The standard EqDeepRx path generates training and validation slots online through the configured Sionna backend; the external DeepRX assets must not be substituted into this run. The implementation audit will record this inventory before launch so that a similarly named external artifact cannot silently change the protocol.

The current README also has a reproducibility-path mismatch: its command writes to `outputs/figure6a_step70000`, while the committed image/JSON are under `results/figure6a_step70000`. The final README will use one explicit runtime/output convention and the checked-in artifact paths will match that convention.

### Figure 6(a) bin coverage

The protocol requests 32,000 validation slots, split evenly between the two pilot settings. With centers `[-5,-3,-1,1,3,5,7,9,11,13] dB` and 1 dB bins, the existing deterministic sample set places 6,625 one-pilot and 6,708 two-pilot samples inside displayed bins, for 13,333 total. The remaining 9,375 and 9,292 samples (18,667 total) are outside the plotted range and are intentionally excluded from displayed bins. This increases per-bin sampling variance; it is not a checkpoint defect and does not invalidate a same-sample 52.5k-versus-70k comparison. The evaluator and README must state this boundary explicitly.

## Paper-fidelity matrix and performance-gap hypotheses

The following is the gate that must pass before a new formal run. A row marked **public** is copied from Sections II--IV and Tables I--II of `D:\EqDeepRxRuns\eq_deeprx_rev.tex`; a row marked **implementation choice** is not allowed to be presented as a paper fact. The audit must produce a machine-readable manifest with one entry per row and a test or source reference for the observed value.

| Area | Required public protocol | Current implementation to verify | Required evidence before training |
| --- | --- | --- | --- |
| Dimensions | `F=192`, `S=14`, `NR=16`, `NT=4` primary model, `B=8`, `E=2` equalizer branches; batch dimension omitted in Table I | `config.py`, `model.py`, `receiver.py` | Shape tests for every boundary and parameter count `115456` (rounds to the paper's 116k) |
| Training layer coverage | One shared model samples `NT=2,3,4`; it is not three separate networks | `TrainingConfig.layer_counts=(2,3,4)` and shared DetectorNN/DemapperNN modules | A seeded batch trace proving all 12 `(layers,pilots,interference)` combinations occur and parameter identities are shared |
| Channel/pilots | UMa training; 1 or 2 orthogonal DMRS symbols, staggered every fourth subcarrier with the Kronecker QPSK sequence reused across selected DMRS symbols; one TX antenna per UE; 0 or 1 interferer | `sionna_system.py` resource grid/topology | Pilot-mask, sequence-reuse, TX-antenna, interference-count, and online-freshness tests |
| DMRS data occupancy | Sionna `KroneckerPilotPattern` reserves the selected DMRS OFDM symbols across the grid; pilots are staggered within those symbols | `signal.py`, `sionna_system.py` | Shape/count regression (`13/14` and `12/14` data-symbol ratios) and direct Sionna mask comparison |
| Distributions | SNR `U(0,45 dB)`, speed `U(0,35 m/s)`, INR `Lognormal(10 dB,5 dB)`, 64-QAM primary path; validation includes UMa/CDL-C/CDL-D/UMi | `config.py`, `sionna_system.py` | Distribution/seed histogram checks and explicit config fingerprint |
| Channel estimation/equalization | Eq. (8) rank-one raw estimate; interpolation; Eq. (9) 24-subcarrier SCM plus complex OAS; Eqs. (2)--(7) RZF/LMMSE with `alpha=1e-4` and unit-gain scaling | `receiver.py` | Independent tensor fixtures against NumPy reference equations, including complex Hermitian/OAS edge cases |
| DenoiseNN | Per RX--TX pilot pair; real/imag; `[64,64,64,2]`, depth 4, subsample `[1,4,2,1]`, frequency-only `13x1`, time mixer after each block | `model.py`, `layers.py` | Shape, receptive-axis, mixer, and per-pair independence tests |
| DetectorNN | Six real inputs, 1x1 projection to 64, four sections; each section has full and 1:8 residual blocks, section shortcut, shared per layer | `model.py`, `layers.py` | Section-count, residual/full-resolution, and shared-weight tests |
| DemapperNN | Four pointwise residual blocks `[32,32,32,B]`, `B=8`, per-layer/shared, positive LLR means bit 0 | `model.py`, `losses.py` | LLR sign, lower-order bit masking, and four-block shape tests |
| Loss/stability | SNR-weighted BCE plus symbol loss from every Detector state, `lambda=1e-5`; mVCL mean target 0, variance target 1, `alpha=1e-5`, divided by channel count | `losses.py`, `training.py` | Equation fixtures, microbatch-vs-full-batch gradient equivalence, and attachment-point manifest |
| Optimizer/schedule | LAMB, initial LR `4.4e-3`, linear decay to zero, effective batch 112, about 70k iterations/about 8M samples | `training.py`, `config.py` | LR/optimizer state trace and exact 112-sample accumulation test |
| Figure 6(a) | CDL-C, 10--15 m/s, one interferer, 64-QAM, 1/2 pilots, 32k slots, realized-SINR binning; Table II places native UMa in the training column and `10--1100 ns` in the validation column, while the exact CDL sampling plumbing is explicitly recorded as an implementation choice; include EqDeepRx, baseline, and known-channel references | `evaluation.py`, CLI | Protocol manifest, five curve keys, layer count 4, and in/out-of-range counts |

The primary Figure 6(a) layer decision is **4**, not 3: Table I uses `N_T=4`, Table II permits 2--4 for the shared training model, and the supplied Figure 6(a) data path is `MUMIMO16x4`. Training must still exercise 2 and 3 layers to verify layer-count invariance; a 3-layer curve may be retained as a diagnostic, but it cannot be labeled as the paper's primary Figure 6(a) result.

The current 52.5k-to-70k comparison establishes only a small, non-zero change under the old three-layer protocol. It does not identify a unique cause. The following hypotheses remain separate until controlled tests confirm them:

1. **Confirmed trajectory defect:** the old recovery reset 20 Demapper optimizer states and permitted incomplete history/non-finite updates. This invalidates the old run as a clean 70k trajectory.
2. **Confirmed protocol defect:** the published evaluator defaulted to 3 layers; it also used a duplicated SINR-center source that must be unified.
3. **Possible data/channel mismatch:** FFT/CP, UMa/CDL delay-spread plumbing, CIR discretization/normalization, antenna array construction, pathloss/shadow-fading switches, and interferer timing are not fully specified by the paper. Table II places native UMa in the training column and publishes `10--1100 ns` in the validation column; it does not specify the exact CDL sampling plumbing, so the sampled CDL default and fixed values are all recorded implementation choices and are compared one at a time. The separate `10--100 ns` reduced-delay-spread BLER experiment is not Figure 6(a).
4. **Possible optimization mismatch:** LAMB beta/epsilon/weight decay/bias correction, AMP boundaries, mVCL attachment, symbol-loss normalization, and TimeMixer channel subset are not fully specified. They must not be changed in the formal paper-aligned run without an explicit sensitivity record.
5. **Statistical explanation:** 18,667 of 32,000 slots fall outside the ten displayed bins, so per-bin variance can make late-training gains look flat. This is measured by the evaluator and is not treated as a training failure.

No plan step may claim that a clean run will be numerically identical to the paper: the paper does not publish all executable details or the authors' random stream. The objective is to (a) remove confirmed implementation errors, (b) prove public-protocol equivalence, and (c) measure which unpublished choices explain the remaining gap without calling an exploratory choice a strict reproduction.

### Performance acceptance rule

The paper gives plotted curves rather than a machine-readable numeric tolerance. Before any candidate is promoted, extract the ten paper marker coordinates from the supplied Figure 6(a) asset (or record that a marker is not digitizable) and store the digitization uncertainty beside the manifest. After the clean 4-layer run, report for each curve and bin the BER, `log10` gap to the digitized marker, slot count, and bootstrap interval. A clean run may be described as matching the paper only where the comparison is within the recorded digitization/statistical uncertainty; otherwise the README must say that it is a protocol-aligned reproduction with a measured numerical gap. A sensitivity candidate may be called an optimization experiment only when it improves the pre-registered same-sample metric across both screening seeds and a confirmation seed; it must never be silently relabeled as the paper's result.

## File Map

- Modify `tests/test_training.py`: add regression tests for checkpoint rejection, finite-step stopping, atomic resume, and exact segmented-vs-uninterrupted behavior.
- Modify `tests/test_evaluation.py`: assert checkpoint provenance fields and in-range/out-of-range sample accounting.
- Modify `tests/test_paper_fidelity.py`: executable assertions for Table I/II dimensions, formulas, layer sharing, loss, optimizer schedule, DMRS occupancy, and the Figure 6(a) protocol manifest.
- Modify `src/eqdeeprx/training.py`: add the checkpoint contract, RNG state capture/restore, finite update guard, anomaly payload, and formal checkpoint schema marker.
- Modify `src/eqdeeprx/config.py` and `src/eqdeeprx/model.py`: expose the unpublished TimeMixer subset `C_s` and AMP boundary as explicit, shape-checked configuration fields.
- Modify `src/eqdeeprx/evaluation.py`: record the checkpoint fingerprint/step supplied by the CLI and explicit out-of-range SINR counts.
- Modify `scripts/train.py`: expose no new training hyperparameters; only pass the existing output/resume paths through the hardened trainer if a small schema option is required.
- Modify `scripts/evaluate_uncoded_ber.py`: validate the completed checkpoint contract before loading a standard evaluation model and pass its provenance to the evaluator.
- Create `scripts/audit_checkpoints.py`: read-only D: audit that prints JSON candidates, rejection reasons, and the highest valid resume point.
- Create `scripts/paper_fidelity_audit.py`: compare the paper manifest with runtime config/source and emit public/implementation-choice/unknown statuses.
- Create `scripts/digitize_paper_figure6a.py`: record paper marker coordinates and digitization uncertainty from the supplied Figure 6(a) asset without modifying the source paper.
- Create `scripts/run_unpublished_sensitivity.py`: deterministic, single-variable short-run harness that never writes the formal checkpoint; `--resume` reuses only matching candidate JSON files and writes an `incomplete` summary while records are missing.
- Create `docs/clean_training_audit.md`: record the current audit and the post-fix audit result.
- Create `docs/paper_fidelity_audit.md`: record every public parameter, implementation choice, test evidence, and sensitivity result.
- Modify `README.md` only after clean training/evaluation verification: objective reproduction scope, exact commands, citation, result image, metric JSON link, and the explicit 13,333/18,667 bin-coverage note.
- Replace repository `checkpoints/eqdeeprx_step70000.pt` and `results/figure6a_step70000/*` only after the new clean artifacts pass every gate.
- Keep runtime outputs outside the repository under `D:\EqDeepRxRuns` until publication is approved.

---

### Task 1: Add failing checkpoint and finite-update tests

**Files:**
- Modify: `tests/test_training.py`
- Modify: `tests/test_evaluation.py`

**Interfaces:**
- Reuse the existing `_tiny_config`, `EqDeepRx`, `OFDMSystem`, and `train_steps` test helpers.
- The trainer must raise `ValueError` for an invalid resume payload and `FloatingPointError` for a non-finite update.

- [x] **Step 1: Write the missing-optimizer-state test.**

Create a one-step tiny checkpoint, remove one entry from `optimizer_state_dict["state"]`, and assert:

```python
with pytest.raises(ValueError, match="optimizer state consistency"):
    train_steps(
        EqDeepRx(config), OFDMSystem(config), config,
        steps=2, batch_size=1, n_layers=2, pilot_count=1, snr_db=12.0,
        seed=19, resume_path=broken,
    )
```

- [x] **Step 2: Write the history-length test.**

Truncate `losses` by one while leaving `next_step` and `history["steps"]` unchanged; assert `ValueError` with `history consistency` before the model or optimizer is loaded.

- [x] **Step 3: Write the non-finite-step test.**

Monkeypatch `eqdeeprx.training.eqdeeprx_loss` with:

```python
def nan_loss(logits, *args, **kwargs):
    return logits.sum() * torch.tensor(float("nan"), device=logits.device)
```

Run one tiny step with an output path and assert `FloatingPointError`, no formal output replacement, an artifact named `broken.nonfinite_step0.pt`, and no history entry claiming a completed step.

- [x] **Step 4: Write the exact resume-equivalence assertion.**

Compare a two-step uninterrupted tiny run with a one-step checkpoint plus one-step resume. Require equal history, equal model tensors, equal optimizer state tensors/counters, equal Python RNG state, and equal saved Torch RNG state.

- [x] **Step 5: Write the evaluation accounting assertions.**

Extend the existing fake Sionna backend test so it asserts `in_range_sample_count + out_of_range_sample_count == validation_samples` and that the per-pilot counts are present in the metrics JSON. Add an assertion that standard evaluation records the checkpoint model fingerprint and `checkpoint_next_step`.

- [x] **Step 6: Run the focused tests and confirm GREEN after the implementation.**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_training.py -k "resume_rejects_inconsistent or nonfinite_step or resume_matches" -q
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation.py -k "figure6a or checkpoint" -q
```

Observed after the implementation: the focused tests pass and reject the legacy payload classes before state loading.

### Task 2: Implement strict checkpoint validation and finite-step handling

**Files:**
- Modify: `src/eqdeeprx/training.py`
- Modify: `scripts/train.py` only for plumbing that remains backward-compatible

**Interfaces:**
- Add `_finite_tensor_tree(value, path) -> list[str]`.
- Add `_checkpoint_consistency_errors(checkpoint, model, optimizer, expected_signature, expected_amp) -> list[str]`.
- Add `_validate_resume_checkpoint(...) -> None`, raising one `ValueError` containing all actionable rejection reasons.
- Add `_make_nonfinite_checkpoint_path(output_path, failed_step) -> Path`.
- Keep existing `train_steps(...)` arguments and the full-run CLI parameters unchanged.

- [x] **Step 1: Validate raw payloads before loading state.**

Require, for `next_step > 0\), all of the following: `next_step == steps == history.steps`; each history list has exactly `next_step` entries; every history value is finite; every model tensor is finite; every trainable model parameter has exactly one optimizer state; each optimizer state contains finite `step`, `exp_avg`, and `exp_avg_sq` values with parameter-matching shapes; every optimizer state counter equals `next_step`; the scaler scale is finite and strictly positive; the config, run signature, batch sizes, AMP flag, and random-state payload match the requested run. Reject explicit `checkpoint_status="nonfinite"`. Accept an old payload without a status marker only when every structural and finite check passes; all new formal saves must write `checkpoint_status="complete"` and `checkpoint_schema_version=1`.

- [x] **Step 2: Capture and restore all deterministic state.**

Save and restore the existing local `random.Random` state plus `torch.get_rng_state()` and, on CUDA, `torch.cuda.get_rng_state_all()`. A replay after an interruption must use the same `seed + step * batch_size + generated` sample seeds and the same model/optimizer/RNG state as the last committed checkpoint.

- [x] **Step 3: Guard loss and gradients before `Lamb.step()`.**

After all microbatch backward passes, call `scaler.unscale_(optimizer)` when AMP is enabled. Check the scalar loss, BER, every gradient, and every accumulated gradient tensor with `torch.isfinite`. If any check fails, write a sibling anomaly payload containing `checkpoint_status="nonfinite"`, `failed_step`, the last formal `next_step`, the requested run signature, the offending diagnostics, and the current finite/non-finite state for analysis; leave the formal output untouched, do not append history, and raise `FloatingPointError`.

- [x] **Step 4: Guard the post-update state.**

Run `scaler.step(optimizer)` and `scaler.update()` only after the pre-step checks. Then verify every model tensor, every optimizer state tensor/counter, the scaler scale, and the proposed history values. Treat a skipped scaler update or any non-finite post-update value as the same hard-stop anomaly. This protects against an overflow generated inside the custom LAMB update rather than in the gradient.

- [x] **Step 5: Save only a complete finite payload atomically.**

Append the history and increment `next_step` only after the post-update checks pass. Build a payload with `checkpoint_status="complete"`, schema version, config/run signature, model/optimizer/scaler state, all RNG states, and history. Use the existing `atomic_torch_save`; never replace the formal file with the `.tmp` file until serialization succeeds. Save every 500 successful steps and at the final step.

- [x] **Step 6: Run the focused tests and the full training test module.**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_training.py -q
```

Expected: all training tests pass, including the new rejection, finite-stop, and segmented-resume tests.

### Task 3: Make the D: audit repeatable and choose a legal starting point

**Files:**
- Create: `scripts/audit_checkpoints.py`
- Create: `docs/clean_training_audit.md`
- Modify: `tests/test_training.py` if the auditor needs a unit-level fixture

**Interfaces:**
- CLI: `scripts/audit_checkpoints.py --root D:\EqDeepRxRuns\checkpoints --steps 70000 --batch-size 112 --microbatch-size 28 --generation-batch-size 2 --seed 2026`.
- Output JSON fields: `file`, `next_step`, `valid`, `reasons`, `history_lengths`, `optimizer_step_counts`, `scaler_scale`, `model_finite`, `optimizer_finite`, and `candidate_rank`.

- [x] **Step 1: Implement the auditor using the same validator as resume.**

Load each `.pt` read-only with `weights_only=False`, never rewrite it, and emit a rejection reason for every failed contract item. Rank valid files by `next_step`; do not rank a file by filename or modification time.

- [x] **Step 2: Run the post-fix audit against all D: files.**

Record the JSON and a human-readable table in `docs/clean_training_audit.md`. The read-only audit already confirms zero valid candidates: the highest-looking `eqdeeprx.pt` fails history length and counter equality; the 15k recovery files fail history/state completeness; the diagnostic files fail counter equality; the NaN files fail finite-history/scaler checks.

- [x] **Step 3: Apply the conditional start rule.**

If the auditor finds a valid candidate, preserve it and resume from that exact file into a new runtime output, e.g. `eqdeeprx_clean70k.pt`. If no candidate is valid, initialize a new model with seed 2026 and do not pass `--resume`. Never use `eqdeeprx.pt` merely because its metadata says 70,000, and never synthesize a missing Demapper state.

- [x] **Step 4: Run all tests before any CUDA long run.**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
git diff --check
```

Expected: zero test failures and no whitespace errors.

### Task 4: Prove paper fidelity and isolate the remaining performance gap

**Files:**
- Create: `tests/test_paper_fidelity.py`
- Create: `scripts/paper_fidelity_audit.py`
- Create: `scripts/run_unpublished_sensitivity.py`
- Create: `docs/paper_fidelity_audit.md`
- Modify: `src/eqdeeprx/evaluation.py`
- Modify: `scripts/evaluate_uncoded_ber.py`
- Modify: `tests/test_evaluation.py`

**Interfaces:**
- `scripts/paper_fidelity_audit.py --paper-source D:\EqDeepRxRuns\eq_deeprx_rev.tex --repo-root . --output D:\EqDeepRxRuns\audits\paper_fidelity.json` emits `public`, `implementation_choice`, `unknown`, `observed`, and `evidence` for every row in the paper-fidelity matrix.
- `scripts/run_unpublished_sensitivity.py --variable NAME --values CSV --screening-steps 2000 --validation-samples 4000 --seeds 2026,2027 --n-layers 4 --output-dir D:\EqDeepRxRuns\sensitivity\NAME --resume` runs only isolated screening jobs, resumes parameter-consistent candidate JSON files, and never reads from or writes to `eqdeeprx_clean70k.pt` or any formal checkpoint.
- `scripts/run_joint_sensitivity.py ... --resume [--max-records N]` runs the six predeclared joint profiles; the optional limit stops only after complete profile/seed records so a paused session can resume without a partial result.
- `evaluate_paper_figure6a(...)` defaults to `n_layers=4` and returns five curve keys: `eqdeeprx_1_pilot`, `eqdeeprx_2_pilots`, `baseline_1_pilot`, `baseline_2_pilots`, and `baseline_known_channel`.

- [x] **Step 1: Write the paper-fidelity tests before changing defaults.**

Add executable assertions for the following exact values: `(192,14,16,4,8,2)`, layer coverage `(2,3,4)`, 64-QAM, pilot spacing 4, RZF `1e-4`, INCM bandwidth 24, DenoiseNN widths/subsampling, DetectorNN 64 channels and four sections, DemapperNN widths ending in 8, parameter count 115456, learning rate `4.4e-3`, effective batch 112, symbol weight `1e-5`, mVCL alpha `1e-5`, and linear LR decay. Also assert that the public Figure 6(a) manifest and evaluator default select four layers; three-layer invocations remain diagnostic only.

- [x] **Step 2: Run the focused tests and verify the corrected defaults.**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_paper_fidelity.py tests/test_evaluation.py -q
```

The focused tests must cover the four-layer default, one SINR-center source, and explicit in/out-of-range counters. Do not weaken them to preserve legacy output.

- [x] **Step 3: Implement the manifest and formula fixtures.**

Parse only the checked-in paper source and store a versioned manifest containing the public values and source line numbers. Add deterministic fixtures for: rank-one channel estimate, complex OAS shrinkage, RZF/LMMSE unit-gain equalization, normalized coordinate maps, nearest-neighbor residual down/up sampling, bit-0-positive LLRs, weighted BCE plus every Detector-state symbol loss, and mVCL population moments. Compare tensor outputs and gradients against small NumPy/PyTorch references; a source comment or a shape assertion alone is not evidence of equation equivalence.

- [x] **Step 3a: Establish the paper performance reference before tuning.**

Run:

```powershell
.\.venv\Scripts\python.exe scripts/digitize_paper_figure6a.py --asset-root D:\EqDeepRxRuns\external\DeepRX\docs\assets --output D:\EqDeepRxRuns\audits\paper_figure6a_reference.json
```

The output must contain the ten x-coordinates, each readable marker's y-coordinate, the extraction method, and an uncertainty or an explicit `unavailable` status. Do not infer missing values from the current reproduction curve.

- [x] **Step 4: Correct and lock the Figure 6(a) protocol.**

Make one configuration constant the source for the ten centers `[-5,-3,-1,1,3,5,7,9,11,13]`, set evaluator/CLI defaults to four layers, and keep 2/3-layer runs explicitly diagnostic. Update progress signatures so a resumed evaluation cannot mix layer counts, curve sets, or checkpoint fingerprints.

- [x] **Step 4a: Lock the verified DMRS occupancy contract.**

Keep the staggered per-layer `pilot_mask` for channel estimation, but derive `data_mask` from the union of the selected pilot OFDM symbols, matching Sionna's `KroneckerPilotPattern`. Add tests that assert the two paths produce the same symbol occupancy and that one/two pilot configurations retain exactly 13/14 and 12/14 data symbols. Do not reinterpret the staggered pilot positions as independently usable data REs; that would contradict the paper's spectral-efficiency definition.

- [x] **Step 5: Build the controlled sensitivity harness.**

For every candidate, hold the model initialization, sample seed stream, 4-layer/1-or-2-pilot coverage, validation sample IDs, number of screening steps, and evaluation bins fixed. Change exactly one implementation choice per job. The baseline is resolved from the formal `paper_config()` value, never from candidate-list order; this is required for `C_s=2` even though the candidate list is ordered `(1,2,4)`. The first screening matrix is:

| Variable | Baseline | Candidate values | Why it is allowed |
| --- | --- | --- | --- |
| TimeMixer selected channels | `2` | `1`, `2`, `4` | `C_s` is not published; shape-safe alternatives test the time-mixing interpretation |
| CDL validation delay spread mode | `uniform_10_1100ns` | `fixed_300ns`, `fixed_100ns`, `fixed_1000ns` | The paper does not publish a numerical CDL validation delay spread; all listed CDL modes are explicit implementation choices |
| UMa delay-spread handling | Sionna native UMa | explicit `10--1100 ns` sampling | Table II's training cell is UMa; the validation cell publishes `10--1100 ns`, but the constructor plumbing is not specified |
| CIR discretization | current sinc/normalization path | Sionna reference path; current path with normalization off | Tests the custom time-domain boundary without changing public network equations |
| pathloss/shadow fading | disabled in current implementation | enabled only if the paper's Sionna scenario reproduces it | Separates topology power scaling from receiver learning |
| interferer timing | current randomized offset | reference-aligned zero/guarded offset | Checks the unreported asynchronous-window convention |
| LAMB details | `(beta1,beta2,eps,wd,bias-correction)=(.9,.999,1e-6,0,True)` | one-at-a-time alternatives, including bias correction off | Never vary more than one optimizer detail per job |
| AMP boundary | bfloat16 convolutions/float32 loss | float32-only diagnostic | Distinguishes numerical safeguard from optimization behavior |
| mVCL attachment | all four Detector states | final state only | Paper gives the formula but not attachment location |
| symbol-loss reduction | current per-sample masked sum | normalized per active RE diagnostic | Checks an unreported reduction convention |
| residual skip projection bias | `false` | `true` | The paper says to project the skip branch with 1x1 but does not publish its bias setting; the default graph remains bias-free |

The formal baseline for `C_s` remains `2` because the paper only says “a small subset”; the field is now explicit so the screening result can be reproduced. The formal CDL path currently samples the published validation range `10--1100 ns` as an explicit implementation baseline; the paper does not publish the exact CDL sampling plumbing. Fixed-delay candidates, Sionna-default pathloss/shadow-fading flags, UMa delay plumbing, CIR normalization, and interferer timing must be reported as implementation choices unless the source or a controlled reference establishes otherwise.

Each job writes JSON with `variable`, `value`, `seed`, `config_fingerprint`, `checkpoint_fingerprint`, training-step count, screening loss/BER, Figure 6(a) BER by bin, runtime, and finite-state status. The harness writes an `incomplete` summary until every expected seed/value record exists; `--resume` reuses only schema-5 records whose run-size/configuration, independent system seed, and validation-plan fingerprint match. The validation-plan fingerprint binds layer count, evaluation batch size, channel model, speed range, pilot counts, and SNR plan while remaining candidate-independent. No candidate may be selected from a single seed or a single plotted point. The 20-step matrix is a screening gate only; it cannot freeze a formal choice.

- [x] **Step 6: Execute screening only after the tests pass and analyze the gap.**

Run the harness with the exact interface above. The original `confirmation_v1` pass used 20 steps for screening seeds `2026/2027` and 2,000 steps only for confirmation seed `2028`; it is therefore not a same-budget three-seed optimum study. The next pass must enumerate every value in the complete candidate matrix, plus the formal baseline where applicable, using the same three seeds and fixed validation sample IDs. `scripts/run_sensitivity_confirmation_supervisor.ps1` now defaults to `-SelectionMode all`; use its dry-run output to verify that values discarded by the short screen (especially `time_mixer_channels=1/4`) are still scheduled. Use a predeclared `--confirmation-steps` value of at least 2,000. For example:

```powershell
.\.venv\Scripts\python.exe scripts/run_unpublished_sensitivity.py `
  --variable cdl_delay_spread_mode `
  --values uniform_10_1100ns,fixed_100ns,fixed_300ns `
  --backend sionna --device cuda --screening-steps 20 --confirmation-steps 2000 `
  --validation-samples 400 --min-in-range-samples 100 `
  --min-in-range-per-pilot 20 --batch-size 8 --microbatch-size 4 `
  --generation-batch-size 2 --output-dir D:\EqDeepRxRuns\sensitivity\confirm_cdl_v1 --resume
```

Repeat that command for each variable with its own isolated output directory; never mix records from different step counts. Compare against the same-sample baseline using pooled mean BER and bootstrap confidence intervals across slots, not visual distance to the paper image. Classify each outcome as `trajectory`, `protocol`, `channel`, `optimizer`, `statistical`, or `unresolved`. A candidate that improves BER but violates a public paper value is an exploratory variant, not the formal reproduction; record it separately and do not silently substitute it. The completed schema-5 run is recorded at `D:\EqDeepRxRuns\sensitivity\confirmation_v1`: 17 variable jobs, 75 finite records, all three seeds represented, but only seed `2028` at the 2,000-step confirmation budget. It is not sufficient to freeze a value until the full candidate matrix has an equal-budget three-seed pass. The Figure 6(a) protocol contains the two EqDeepRx curves and the three specified baseline/reference curves.

- [ ] **Step 7: Freeze the formal configuration and document the decision.**

Write `docs/paper_fidelity_audit.md` with the manifest, test outputs, every candidate and rejection reason, and the exact frozen configuration. The one-variable matrix currently confirms fixed 100 ns, fixed 300 ns, and LAMB bias correction off. Before selecting a combined exploratory profile, run `scripts/run_joint_sensitivity.py` for the six predeclared profiles and all three seeds, then write `D:\EqDeepRxRuns\audits\sensitivity_freeze.json` with `scripts/freeze_sensitivity.py`. The formal run may proceed only when the joint report is complete, all public rows are `matched`, all unknown rows have a recorded choice and rationale, and no selected choice is based solely on an unreplicated improvement. If no candidate explains the gap, state that the remaining difference is attributable to unavailable author-private details rather than inventing a claim of paper-level numerical equivalence. The finite matrix is reported as the best validated tested configuration, never as a mathematical global optimum.

- [x] **Step 8: Run all focused and repository tests before a formal checkpoint is touched.**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_paper_fidelity.py tests/test_training.py tests/test_evaluation.py -q
```

Expected: all new contract/protocol tests pass, while all existing tests remain green.

### Task 5: Verify preflight and run the formal 70,000 successful steps

**Files:**
- Runtime create: `D:\EqDeepRxRuns\checkpoints\eqdeeprx_clean70k.pt`
- Runtime create: `D:\EqDeepRxRuns\train_long.stdout.log` (after archiving the legacy log)
- Runtime create: `D:\EqDeepRxRuns\train_long.stderr.log` (after archiving the legacy log)

- [ ] **Step 1: Re-run the standard preflight without changing formal parameters.**

```powershell
.\.venv\Scripts\python.exe scripts/preflight.py --device cuda --microbatch-size 28 --generation-batch-size 2 --output D:\EqDeepRxRuns\outputs\preflight_clean70k.json
```

Require the paper config fingerprint, complete 12/12 layer/pilot/interference coverage, finite standard loss and gradients, approved microbatch 28, generation batch 2, and CUDA availability. Record the estimated runtime and `long_training_ready` field, but treat the repository's eight-day value as advisory because the user waived that convenience threshold.

- [ ] **Step 2: Launch only after the preflight and audit gates pass.**

Use the unchanged formal command and a new output path. If the audit reports a valid candidate, append `--resume` followed by that exact audited file path; otherwise omit `--resume`:

```powershell
Start-Process -WindowStyle Hidden -FilePath .\.venv\Scripts\python.exe -ArgumentList @(
  'scripts/train.py','--confirm-full-run','--steps','70000',
  '--batch-size','112','--microbatch-size','28','--generation-batch-size','2',
  '--device','cuda','--seed','2026',
  '--preflight-report','D:\EqDeepRxRuns\outputs\preflight_clean70k.json',
  '--save-every','500',
  '--output','D:\EqDeepRxRuns\checkpoints\eqdeeprx_clean70k.pt'
) -RedirectStandardOutput D:\EqDeepRxRuns\train_long.stdout.log -RedirectStandardError D:\EqDeepRxRuns\train_long.stderr.log -PassThru
```

Before launch, preserve the existing legacy `train_long.stdout.log` and `train_long.stderr.log` under a timestamped read-only archive name. The active run uses the exact `train_long.*` paths required by the monitor; each report must identify which log generation it scanned.

After the process ID is confirmed, reactivate the existing `eqdeeprx-long-training-monitor` heartbeat rather than creating a duplicate. Each two-hour report must include the actual `next_step`, scalar `history.steps`, lengths of `losses`/`bers`/`learning_rates`, latest finite loss and BER, process state, complete scans of both active logs for Traceback/CUDA OOM/NaN/Inf/other errors, GPU utilization/memory/temperature, C: and D: free space, and the checkpoint's last-write timestamp. Do not launch evaluation curves during this phase.

- [ ] **Step 3: Handle ordinary interruption without changing the run.**

Stop only at a safe process boundary when possible. If termination occurs between saves, the old atomic formal file remains authoritative; restart the identical command with `--resume D:\EqDeepRxRuns\checkpoints\eqdeeprx_clean70k.pt`. The trainer replays the unsaved step range from the saved RNG state, so `next_step` and history count never jump. Delete only stale `.tmp` files after confirming the formal file is intact.

- [ ] **Step 4: Handle process exit and numerical anomalies differently.**

If the process exits without a non-finite anomaly and the latest formal checkpoint passes the validator with `next_step < 70000`, restart with the exact same arguments and `--resume`. If a `nonfinite_step*.pt` artifact or a log containing NaN/Inf appears, do not restart or overwrite; preserve the anomaly, report `failed_step` and the last valid checkpoint, and investigate before any parameter change.

- [ ] **Step 5: Verify the completed checkpoint before evaluation.**

Require `next_step=steps=history.steps=70000`, all three history lengths 70000, finite history/model/optimizer/scaler values, one optimizer state per trainable parameter, every optimizer state counter exactly 70000, matching config/run signature, saved RNG states, `checkpoint_status="complete"`, and an atomic final write timestamp. A metadata-only claim or a finite model tensor alone is not sufficient. Stop the monitor only after the final checkpoint and final evaluation provenance have been recorded.

### Task 6: Run and provenance-stamp Figure 6(a) immediately after clean completion

**Files:**
- Modify: `src/eqdeeprx/evaluation.py`
- Modify: `scripts/evaluate_uncoded_ber.py`
- Modify: `tests/test_evaluation.py`
- Runtime create: `D:\EqDeepRxRuns\outputs\figure6a_clean70k\*`

- [ ] **Step 1: Validate the checkpoint in the evaluator.**

Before constructing the standard Sionna backend, load and validate the checkpoint. Refuse evaluation of a non-complete or inconsistent standard checkpoint. Compute the same model fingerprint used by the progress signature and record `checkpoint_next_step`, `checkpoint_status`, and `checkpoint_model_fingerprint` in `figure6a_metrics.json`.

- [ ] **Step 2: Track in-range and out-of-range realized-SINR samples.**

Extend progress and metrics with per-pilot `in_range_sample_count` and `out_of_range_sample_count`, while retaining the ten displayed bin counts. Increment the out-of-range counter whenever `bin_index` is outside `[0, len(sinr_centers))`; do not silently imply all 32,000 samples contribute to plotted points.

- [ ] **Step 3: Run the fixed protocol without changing it.**

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_uncoded_ber.py --checkpoint D:\EqDeepRxRuns\checkpoints\eqdeeprx_clean70k.pt --validation-samples 32000 --evaluation-batch-size 2 --n-layers 4 --seed 2026 --device cuda --resume --output-dir D:\EqDeepRxRuns\outputs\figure6a_clean70k
```

Require CDL-C, 10-15 m/s, one interferer, 64-QAM, both pilot settings, the ten 2 dB-spaced displayed centers, all five curve keys, finite curves, and a progress signature matching the clean checkpoint fingerprint. The 4-layer command is the formal Figure 6(a) run; any 3-layer output is diagnostic only.

- [ ] **Step 4: Verify images and machine-readable output.**

Check that all BER values are finite where a bin has samples, the image has non-zero pixels and the expected dimensions, `checkpoint_next_step == 70000`, the fingerprint matches the verified checkpoint, the five curve keys are present, and the counts are 13,333 in range and 18,667 out of range in total (6,625/9,375 for one pilot and 6,708/9,292 for two pilots for the current deterministic protocol).

### Task 7: Replace published artifacts and perform final verification

**Files:**
- Modify: `README.md`
- Do not publish: the rejected `checkpoints/eqdeeprx_step70000.pt` or the rejected `results/figure6a_step70000/*` files. Keep new runtime artifacts under `D:\EqDeepRxRuns` until they pass the clean-run audit.

- [ ] **Step 1: Write objective README content.**

State only the reproduced scope, environment/install command, preflight/train/resume/evaluation commands, paper citation, and the explicit displayed-bin coverage note. Do not claim bit-identical private-author output, do not describe the user's requests, and do not call the old inconsistent checkpoint a 70k reproduction. A clean plot and metrics file may be published only after a new checkpoint passes the contract; the current repository intentionally contains no result image or runtime JSON.

- [ ] **Step 2: Keep runtime artifacts outside the repository until the clean-run gate passes.**

Write the clean checkpoint and Figure 6(a) JSON/PNG to `D:\EqDeepRxRuns` and verify their SHA-256 hashes there. Do not copy logs, temporary progress files, diagnostic checkpoints, private/static validation dumps, or unrelated test outputs into the repository. The publication decision is a separate later change.

- [ ] **Step 3: Run the final verification set.**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
git diff --check
.\.venv\Scripts\python.exe scripts/audit_checkpoints.py --root D:\EqDeepRxRuns\checkpoints --steps 70000 --batch-size 112 --microbatch-size 28 --generation-batch-size 2 --seed 2026
```

Read the complete outputs, verify the clean checkpoint and metrics independently, inspect the README diff, and confirm no invalid artifact was staged.

- [ ] **Step 4: Commit and push only after the user approves publication.**

Commit the verified code/tests/docs/results, push `main` to `origin`, and verify the remote commit SHA. The current published commit remains untouched until this gate; no GitHub write occurs during the planning/review phase.

## Acceptance Checklist

- [ ] The current D: audit and the corrected root cause are documented.
- [ ] A legal existing resume point is used if and only if it passes every contract check; otherwise the run starts cleanly.
- [ ] No non-finite update advances `next_step`, appends history, or replaces the formal checkpoint.
- [ ] An interruption resumes from the last atomic valid checkpoint with unchanged formal arguments and no fabricated steps.
- [ ] The final checkpoint has exactly 70,000 finite history entries and 70,000 state counters for every trainable parameter.
- [ ] Figure 6(a) records the exact clean checkpoint fingerprint and step, finite metrics, and explicit in/out-of-range sample counts.
- [ ] README is objective and concise, cites the EqDeepRx paper, and contains the verified result image and reproduction commands.
- [ ] Full tests and final audits pass before any commit or push.

## Execution Gate

This document is the recorded review and execution plan. Production changes already made are limited to the listed gates; the formal 70,000-step run and any result publication remain blocked until the gates and controlled checks below pass.

## Answers to the Review Questions

These are the current evidence-backed answers, separate from hypotheses about the remaining performance gap:

1. **Was the legacy curve loaded from a file labeled 70,000?** Yes. The evaluator fingerprint matches the tensors in the legacy `next_step=70000` file, not a 52,500 file. That proves the input file used by the old plot, but it does not prove a clean 70,000-update trajectory.
2. **Was that trajectory a valid clean 70,000-step run?** No. The audit found 69,666 history entries, 60 optimizer counters at 69,666, 20 Demapper counters at 55,000, missing RNG state, and no complete-checkpoint marker. The file is rejected; it will not be repaired or relabeled.
3. **Did the Demapper counter alone prove that it stopped at global step 55,000?** No. It proves a partial optimizer-state reset/recovery history, not the exact first bad internal update. The surviving files do not identify one unique overflow tensor. The new contract rejects this class of state before it can affect a formal run.
4. **Can the old plateau be attributed only to that reset?** No. The reset is a confirmed trajectory defect and can change later updates, but there is no clean counterfactual checkpoint that isolates its quantitative effect. The plateau is therefore re-tested by the controlled matrix rather than presented as a proven single cause.
5. **How many layers are used?** The shared training model covers 2, 3, and 4 layers. The primary Figure 6(a) result is four layers. Two- and three-layer evaluations remain diagnostics only.
6. **Are the published formulas and public settings aligned?** The paper manifest reports 30/30 public rows matched, including the UMa training delay-spread row; the CDL validation delay spread is not mislabeled as a public value. Executable tests cover raw channel estimation, interpolation, complex OAS, RZF/LMMSE unit-gain equalization, residual block order/subsampling, weighted BCE and symbol loss, mVCL moments, DMRS occupancy, shared layer handling, and the four-layer evaluation protocol.
7. **What remains unknown?** The paper does not expose the exact random stream, TimeMixer subset size, several channel plumbing details, LAMB internals, AMP boundary, mVCL attachment, or symbol-loss reduction. These are recorded as implementation choices and tested one at a time. No exploratory improvement may replace a public paper value or be called the strict reproduction.
8. **Can the paper's exact numerical curve be guaranteed before a clean run?** No. The paper supplies plotted points but not all author-private data/streams. The acceptance criterion is a valid protocol-aligned result with provenance and an honest measured comparison, not an unsupported claim of bit identity.

## Current execution status (2026-09-25)

- [x] Legacy D: checkpoints audited read-only; no valid continuation selected.
- [x] Resume contract, RNG capture/restore, atomic save, and non-finite anomaly preservation implemented and covered by tests.
- [x] Primary Figure 6(a) protocol locked to four layers and one SINR-center source; in/out-of-range counts are recorded.
- [x] Public paper manifest implemented and checked against `D:\EqDeepRxRuns\eq_deeprx_rev.tex` (30/30 public rows matched, including published INR and one-TX-antenna rows).
- [x] Unpublished configuration fields implemented without changing formal defaults: TimeMixer channels, residual projection bias, sampled-vs-fixed CDL delay spread, UMa delay mode, CIR mode/normalization, pathloss/shadow fading, interferer timing, LAMB details, AMP dtype, mVCL attachment, and symbol-loss reduction.
- [x] Fixed-sample sensitivity harness, resumable candidate files, explicit incomplete summaries, and reference-asset availability tool implemented; tiny Sionna coverage smoke test passed and exploratory output is isolated from formal checkpoints.
- [x] Table II delay-spread columns are explicit: native UMa is the training entry and `10--1100 ns` is the validation entry; sampled and fixed CDL modes are recorded as implementation choices, and the separate `10--100 ns` experiment is excluded from Figure 6(a); focused tests pass.
- [x] Fast-backend interference timing is now explicit and tested: `random_symbol`, `random_sample`, and `zero` use distinct, documented time-axis offsets in the smoke backend. This fixes a configuration-plumbing defect without changing the formal Sionna default.
- [x] Historical regression evidence was recorded under an earlier environment. The current repository-environment regression is `156 passed` with no skips in `.venv`; `git diff --check` is also clean. The fast TimeMixer probe completed 9/9 records, but every record has zero displayed-SINR coverage, so the run is retained as invalid smoke evidence and no candidate is frozen from it.
- [x] Fresh schema-5 Sionna sensitivity matrix completed in `confirmation_v1`: 17/17 jobs, 75/75 finite records, seeds `2026/2027/2028`, and no formal checkpoint touched.
- [x] The short-screen winners and confirmation-seed records are recorded separately from the formal baseline; the screening seeds used 20 steps, so this is not yet an equal-budget three-seed confirmation.
- [x] Confirmation supervisor corrected: default `-SelectionMode all` enumerates every candidate-matrix value, passes one equal confirmation budget to seeds 2026/2027/2028, and supports `-DryRun`; `-SelectionMode screening-winners` remains an explicit legacy mode.
- [ ] Complete the six-profile joint Sionna gate and write the final freeze manifest before long training. The user paused this gate after the first complete record on 2026-09-28; no later profile was started.
- [x] One joint preflight record completed before the pause: `formal_baseline`, Sionna, four layers, seed `2026`, 2,000 steps, finite state, 161/400 in-range validation samples, and `formal_checkpoint_touched=false`. Its pooled EqDeepRx BER was `0.209386304607957` versus `0.199265762471` for the traditional two-pilot/one-pilot baseline counters in the same validation run; this is an under-trained diagnostic, not a candidate improvement.
- [x] Joint runner now supports `--max-records` for a clean, resumable stop at a completed profile/seed boundary. The partial `joint_summary.json` remains `status=incomplete` and cannot produce a launch freeze.
- [x] Standard CUDA preflight has been rerun without competing GPU jobs: 12/12 configurations, finite loss/gradients, 4.0 GiB headroom, and an estimated 8.047 days at the approved batch settings. The user has waived the advisory eight-day convenience threshold; the estimate remains a required status field but is no longer a launch blocker. Numerical-safety and checkpoint-contract gates are unchanged.
- [ ] Start a new clean formal run only if no audited checkpoint is valid; resume only from a checkpoint that passes every contract check.
- [ ] Verify exactly 70,000 successful updates, then immediately run the four-layer Figure 6(a) evaluator and provenance checks.

### Paused joint-validation analysis (2026-09-28)

The first joint record is a valid protocol smoke result, not evidence for a
global optimum: it has one seed, only 2,000 optimizer steps, 161 displayed-bin
samples out of 400, and the learned curves remain above the conventional
baseline on the pooled metric. At low SINR the EqDeepRx curve is worse than the
same-run conventional baseline; it becomes slightly better only in the highest
bins, which is expected from an under-trained model and is not a parameter
diagnosis. No unpublished value is changed on the basis of this record. The
next session should first run the full-candidate equal-budget confirmation pass,
then resume the exact six-profile command with `--resume` and complete all 18
profile/seed records. Only after every record is finite, coverage-valid, and
better than the formal baseline under the preregistered three-seed rule may the
strict freeze validator select an exploratory profile. The long 70k run remains
gated by that report and by explicit user approval.

## Requirement Traceability Before Long-Run Approval

This section records the direct answer to each review point raised during the
pre-flight audit. It is deliberately evidence-based; an item marked pending
is a launch gate, not an assumption.

| Review point | Confirmed answer and evidence | Status before formal 70k run |
| --- | --- | --- |
| Does the old curve really load the file labeled 70k? | Yes. The evaluator fingerprint matches the tensors in the old `next_step=70000` file. That proves file provenance only; it does not prove 70,000 valid updates. | Resolved; old artifact rejected |
| Why did the old file fail? | Its histories contain 69,666 entries, 60 optimizer counters are 69,666, 20 Demapper counters are 55,000, and complete RNG/checkpoint metadata are absent. The exact first overflowing internal tensor cannot be recovered from the surviving payloads. | Resolved root cause class |
| Can the 52.5k-to-70k plateau be attributed only to Demapper reset? | No. The reset is a confirmed trajectory defect, but no clean counterfactual isolates its numerical contribution. The new finite-step contract removes that defect; the controlled matrix tests the remaining plateau hypotheses. | Resolved limitation; quantitative cause pending |
| How many layers are required? | Training samples one shared model over 2, 3, and 4 layers. The primary Figure 6(a) protocol is 4 layers, as required by the `N_T=4` Table I dimensions and the supplied MUMIMO16x4 path. A 3-layer curve is diagnostic only. | Resolved and locked in code/tests |
| Were the paper dimensions and formulas checked? | The source-backed manifest has 30/30 public rows matched (`F=192`, `S=14`, `N_R=16`, `N_T=4`, `B=8`, `E=2`, 30 kHz, distributions, architecture widths, receiver constants, loss weights, batch/LR/decay). Executable tests cover the raw estimate, interpolation, OAS, RZF/LMMSE, residual blocks, demapper sign, symbol loss, mVCL, DMRS occupancy, and layer sharing. | Public gate passed |
| What is known about `C_s` and other unreported details? | `C_s=2`, residual projection bias (default `false`), channel plumbing, CIR mode/normalization, pathloss/shadow fading, interferer timing, LAMB internals, AMP boundary, mVCL attachment, and symbol-loss reduction are implementation choices. They are exposed in the config and tested one variable at a time. | Matrix/confirmation evidence required |
| Is a complete static dataset available on D:? | No complete EqDeepRx training/validation dump was found. The formal path generates samples online through Sionna. The separate `external\\DeepRX` reference tree is not substituted into this protocol. | Resolved; online generation retained |
| How is interruption handled? | Only an atomic `checkpoint_status=complete` payload is resumable. Resume requires matching config/signature, exact history lengths, every optimizer counter equal to `next_step`, finite scaler/model/optimizer values, and saved Python/Torch/CUDA RNG states. A non-finite step is written beside the last valid file and never overwrites it. | Implemented and unit-tested |
| Are Figure 6(a) bin counts interpreted correctly? | Yes. The fixed 32,000 slots yield 13,333 in-range samples and 18,667 out-of-range samples under the ten displayed centers; both counts are written to metrics and README. | Resolved and instrumented |
| Can paper-level numerical equality be promised now? | No. The paper does not publish author-private random streams or all executable channel/optimizer details, and the supplied plot has no machine-readable coordinates. We will report a measured, provenance-stamped comparison and will not claim bit identity without evidence. | Explicit non-claim |
| May the 70k run start now? | No. The latest preflight has finite loss/gradients and complete configuration coverage, and the user waived the advisory runtime threshold; the remaining blocker is the user's explicit approval plus completion of the controlled sensitivity gate. | Blocked by explicit user gate and sensitivity evidence |
