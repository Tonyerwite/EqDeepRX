# EqDeepRx Reproduction

This repository contains an independent PyTorch/Sionna implementation of *EqDeepRx: Learning a Scalable and Interference Mitigating MIMO Receiver*. The reproducible scope is the paper's 64-QAM receiver and uncoded BER path. LDPC decoding, other decoder-dependent experiments, and DenoiseNN-only ablations are outside the scope.

## Implementation

- Sionna 2.1 time-domain TR 38.901 channels with online UMa training and CDL-C Figure 6(a) evaluation.
- DenoiseNN, conventional RZF/LMMSE equalization, shared per-layer DetectorNN, and DemapperNN.
- Paper-aligned equations, loss terms, LAMB schedule, effective batch size 112, and shared training coverage for 2, 3, and 4 layers.
- Primary Figure 6(a) evaluation uses four layers, one interferer, 10--15 m/s, one- and two-pilot cases, realized-SINR binning, and 32,000 validation slots.

The public paper audit covers 30/30 source-backed configuration rows. Unpublished implementation choices are exposed in the configuration; the equal-budget sensitivity matrix and freeze gate are defined in [`docs/superpowers/plans/2026-09-28-unpublished-config-search.md`](docs/superpowers/plans/2026-09-28-unpublished-config-search.md) and have not yet frozen an exploratory replacement for the formal defaults.

## Environment

The standard path is Windows-oriented and uses Python 3.12, PyTorch 2.9.1+cu128, and Sionna 2.1.0. Create a virtual environment and install the standard dependencies with:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-standard.txt
```

Run the CPU smoke preflight and the repository tests with:

```powershell
.\.venv\Scripts\python.exe scripts/preflight.py --tiny --device cpu
.\.venv\Scripts\python.exe -m pytest -q
```

## Sensitivity confirmation

Before a formal run, the complete unpublished-configuration matrix can be checked without touching a formal checkpoint. The supervisor's dry run only prints the planned jobs:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/run_sensitivity_confirmation_supervisor.ps1 `
  -ScreeningDir D:\EqDeepRxRuns\sensitivity\matrix_screen5_full_v2 `
  -RootOutputDir D:\EqDeepRxRuns\sensitivity\confirmation_v2 `
  -SelectionMode all -DryRun
```

After review, run the full three-seed Sionna confirmation into the D: runtime directory:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/run_sensitivity_confirmation_supervisor.ps1 `
  -ScreeningDir D:\EqDeepRxRuns\sensitivity\matrix_screen5_full_v2 `
  -RootOutputDir D:\EqDeepRxRuns\sensitivity\confirmation_v2 `
  -SelectionMode all -ConfirmationSteps 2000
```

The jobs use seeds 2026, 2027, and 2028 at the same confirmation-step budget, four layers, fixed validation samples, and no formal checkpoint. The complete matrix contains 39 candidate values and therefore 117 seed records. The freeze audit must report `ready_for_long_training` before the formal path is started.

## Clean 70,000-step run

After the sensitivity and a fresh CUDA preflight report with `long_training_ready=true` are complete, run a new formal checkpoint under `D:\EqDeepRxRuns` rather than reusing an older artifact. The repository does not contain a validated result image or checkpoint yet:

```powershell
.\.venv\Scripts\python.exe scripts/preflight.py `
  --device cuda --microbatch-size 28 --generation-batch-size 2 `
  --output D:\EqDeepRxRuns\audits\preflight_clean70k.json

.\.venv\Scripts\python.exe scripts/train.py --confirm-full-run `
  --steps 70000 --batch-size 112 --microbatch-size 28 `
  --generation-batch-size 2 --device cuda --seed 2026 `
  --preflight-report D:\EqDeepRxRuns\audits\preflight_clean70k.json `
  --save-every 500 --output D:\EqDeepRxRuns\checkpoints\eqdeeprx_clean70k.pt
```

If the process is interrupted, resume with the identical arguments and the audited checkpoint path:

```powershell
.\.venv\Scripts\python.exe scripts/train.py `
  --confirm-full-run --steps 70000 --batch-size 112 --microbatch-size 28 `
  --generation-batch-size 2 --device cuda --seed 2026 `
  --preflight-report D:\EqDeepRxRuns\audits\preflight_clean70k.json `
  --resume D:\EqDeepRxRuns\checkpoints\eqdeeprx_clean70k.pt `
  --save-every 500 --output D:\EqDeepRxRuns\checkpoints\eqdeeprx_clean70k.pt
```

The checkpoint validator requires matching `next_step`, `steps`, and history lengths, finite model/optimizer/scaler state, synchronized optimizer counters, saved RNG state, and `checkpoint_status=complete`.

## Figure 6(a) evaluation

Only a validated completed checkpoint may be evaluated:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_uncoded_ber.py `
  --checkpoint D:\EqDeepRxRuns\checkpoints\eqdeeprx_clean70k.pt `
  --validation-samples 32000 --evaluation-batch-size 2 --n-layers 4 `
  --seed 2026 --device cuda --resume `
  --output-dir D:\EqDeepRxRuns\outputs\figure6a_clean70k
```

Validation slots are generated online. With the fixed protocol, 13,333 of 32,000 slots fall inside the ten displayed SINR bins; 18,667 are recorded as out of range and excluded from plotted bins. The evaluator records both counts and checkpoint provenance in `figure6a_metrics.json`.

No checkpoint, plot, or metric file from the rejected legacy 70,000-step trajectory is included in this repository. A result is publishable only after the clean checkpoint and the four-layer evaluation pass the independent audits.

## Reproduction boundary

The implementation is an independent, protocol-aligned reproduction. The paper does not publish every executable detail or the authors' private random stream, so bit-identical numerical output cannot be inferred from the paper alone. Unpublished choices and their evidence are documented in [`docs/paper_fidelity_audit.md`](docs/paper_fidelity_audit.md) and [`docs/clean_training_audit.md`](docs/clean_training_audit.md).

## Citation

Mikko Honkala, Dani Korpi, Elias Raninen, and Janne M. J. Huttunen, “EqDeepRx: Learning a Scalable and Interference Mitigating MIMO Receiver,” arXiv:2602.11834v2, 2026. <https://arxiv.org/abs/2602.11834v2>
