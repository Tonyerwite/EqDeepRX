# Unpublished Configuration Search and Freeze Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine the best validated configuration within the declared unpublished-parameter matrix before any clean 70,000-step run, without changing the paper-public protocol or claiming a mathematical global optimum.

**Architecture:** Use fixed-sample Sionna experiments with the same four-layer Figure 6(a) protocol, independent system seeds, and no formal checkpoint writes. First evaluate every candidate value at the same 2,000-step budget for seeds 2026/2027/2028; then test only candidates that pass the equal-budget per-seed rule in explicitly declared joint profiles and a longer confirmation budget. Freeze an exploratory profile only when the machine-readable gate is complete; otherwise retain the paper-aligned defaults.

**Tech Stack:** Python 3.12, PyTorch 2.9.1+cu128, Sionna 2.1.0, CUDA, PowerShell, pytest.

## Global Constraints

- Do not start `scripts/train.py`, the formal evaluator, or a 70,000-step run while this plan is being executed.
- Do not generate or publish a DenoiseNN-only curve; the Figure 6(a) protocol has four layers and exactly five curves: two EqDeepRx curves and three baseline/reference curves.
- Keep all public paper values unchanged: `F=192`, `S=14`, `N_R=16`, primary `N_T=4`, `B=8`, `E=2`, shared training coverage 2/3/4, receiver equations, loss equations, batch/LR schedule, and validation protocol.
- Every sensitivity record must use Sionna, seeds `2026,2027,2028`, four layers, validation batch size 2, 400 validation samples, the same displayed SINR centers, and the same in-range coverage thresholds (100 total and 20 per pilot).
- A sensitivity result is exploratory. It may be called the best tested finite candidate only; it may not be called the authors' exact result or a mathematical global optimum.
- Any non-finite or incomplete record blocks freezing. Formal checkpoints must remain untouched.

## Evidence Already Collected

The completed record at `D:\EqDeepRxRuns\sensitivity\joint_v1\formal_baseline_seed2026.json` is a valid short diagnostic: 2,000 steps, latest loss `1.3483670353889465`, latest training BER `0.10590486321598291`, pooled validation EqDeepRx BER `0.20938630460795707`, conventional baseline BER `0.199265762471`, 161/400 displayed-bin samples, 239/400 outside the plotted range, finite state, and `formal_checkpoint_touched=false`. Its low-SINR bins are worse than the conventional baseline while its highest-SINR bins are slightly better. This is consistent with an under-trained model and does not identify a parameter defect.

The schema-5 one-variable directory `D:\EqDeepRxRuns\sensitivity\confirmation_v1` contains 75 finite records. Its screening seeds used 20 steps; only seed 2028 used the 2,000-step confirmation budget. The provisional labels for fixed 100 ns, fixed 300 ns, and LAMB bias correction off therefore cannot be interpreted as three-seed, same-budget confirmation. The old confirmation supervisor also scheduled only short-screen winners, which omitted candidates such as `time_mixer_channels=1` and `4`. `scripts/run_sensitivity_confirmation_supervisor.ps1` now defaults to `-SelectionMode all` and has a tested `-DryRun` path; no worker is left running.

## Candidate Matrix

The complete matrix in `scripts/run_unpublished_sensitivity.py` has 17 variables and 39 values, hence 117 candidate/seed records at the equal-budget stage:

| Variable | Values |
| --- | --- |
| `time_mixer_channels` | `1, 2, 4` |
| `cdl_delay_spread_mode` | `uniform_10_1100ns, fixed_300ns, fixed_100ns, fixed_1000ns` |
| `uma_delay_spread_mode` | `native, uniform_10_1100ns` |
| `cir_discretization` | `sinc, nearest` |
| `cir_normalization` | `true, false` |
| `pathloss` | `false, true` |
| `shadow_fading` | `false, true` |
| `interferer_timing` | `random_symbol, zero, random_sample` |
| `lamb_beta1` | `0.9, 0.8` |
| `lamb_beta2` | `0.999, 0.99` |
| `lamb_eps` | `1e-6, 1e-8` |
| `weight_decay` | `0.0, 1e-4` |
| `lamb_bias_correction` | `true, false` |
| `amp_dtype` | `bfloat16, float32` |
| `vcl_attachment` | `all, final, none` |
| `symbol_loss_reduction` | `sum, mean_active` |
| `residual_projection_bias` | `false, true` |

### Task 1: Verify the stopped state and the corrected job planner

**Files:**
- Verify: `scripts/run_sensitivity_confirmation_supervisor.ps1`
- Verify: `tests/test_sensitivity_confirmation_supervisor.py`
- Verify: `D:\EqDeepRxRuns\sensitivity\joint_v1\joint_summary.json`

- [x] **Step 1: Confirm no EqDeepRx worker is running.**

Run:

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match 'EqDeepRX|eqdeeprx|run_unpublished_sensitivity|run_joint_sensitivity|scripts\\train.py' } |
  Select-Object ProcessId,Name,CommandLine
```

Expected: no project sensitivity worker and no formal `scripts/train.py` process. Do not terminate unrelated project processes.

- [x] **Step 2: Verify the planner without starting a worker.**

Run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/run_sensitivity_confirmation_supervisor.ps1 `
  -ScreeningDir D:\EqDeepRxRuns\sensitivity\matrix_screen5_full_v2 `
  -RootOutputDir D:\EqDeepRxRuns\sensitivity\confirmation_dryrun_check `
  -SelectionMode all -DryRun
```

Expected: 17 JSON job entries and values containing `1,2,4` for `time_mixer_channels`; no `run_unpublished_sensitivity.py` process is created. The temporary dry-run directory is not a repository artifact.

- [x] **Step 3: Run the focused regression.**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_sensitivity_confirmation_supervisor.py -q
```

Expected: `1 passed`.

### Task 2: Complete equal-budget confirmation for every candidate value

**Files:**
- Runtime create: `D:\EqDeepRxRuns\sensitivity\confirmation_v2\*`
- Read-only source: `scripts/run_unpublished_sensitivity.py`
- Read-only source: `scripts/run_sensitivity_confirmation_supervisor.ps1`

- [ ] **Step 1: Run the full-candidate supervisor only after the user asks to resume.**

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/run_sensitivity_confirmation_supervisor.ps1 `
  -ScreeningDir D:\EqDeepRxRuns\sensitivity\matrix_screen5_full_v2 `
  -RootOutputDir D:\EqDeepRxRuns\sensitivity\confirmation_v2 `
  -SelectionMode all -ConfirmationSteps 2000
```

The supervisor must run variables sequentially, pass `--values` for every matrix value, use seeds `2026,2027,2028` at the same `--confirmation-steps` budget (the script passes that value to both step arguments), and never write a formal checkpoint. A candidate that errors must remain an explicit error record; it must not be silently removed from the expected matrix.

- [ ] **Step 2: Validate the record count and provenance.**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_audit_tools.py tests/test_joint_sensitivity.py tests/test_freeze_sensitivity.py -q
```

Then inspect `D:\EqDeepRxRuns\sensitivity\confirmation_v2\*\sensitivity_summary.json` and require 39 values, three seeds per value, `training_steps=2000` for all three seeds, finite state, valid coverage, matching validation fingerprints, and `formal_checkpoint_touched=false`. The expected complete record count is `117`; any missing or errored candidate blocks the freeze gate.

- [ ] **Step 3: Rank candidates by a paired, predeclared metric.**

For each variable/value, compute pooled EqDeepRx BER over all populated bins for each seed, the per-seed delta against that variable's formal baseline, the worst-seed delta, and a bootstrap interval over validation slots. A candidate is eligible only when all three per-seed deltas improve, coverage is valid for both pilots, and the interval does not rely on an empty bin. Keep public values and exploratory values in separate columns.

### Task 3: Test interactions without overfitting the short run

**Files:**
- Modify only if the confirmed matrix requires new profile names: `scripts/run_joint_sensitivity.py`
- Runtime create: `D:\EqDeepRxRuns\sensitivity\joint_v1\*`
- Runtime create for the extended pass: `D:\EqDeepRxRuns\sensitivity\joint_v2_10k\*`

- [ ] **Step 1: Finish the existing six-profile 2,000-step gate.**

After Task 2 identifies the eligible values, resume the predeclared profiles with the same six names and no changed public settings:

```powershell
.\.venv\Scripts\python.exe scripts/run_joint_sensitivity.py `
  --profiles formal_baseline,fixed100,fixed300,bias_no_correction,fixed100_bias_no_correction,fixed300_bias_no_correction `
  --steps 2000 --seeds 2026,2027,2028 --backend sionna --device cuda `
  --batch-size 8 --microbatch-size 4 --generation-batch-size 2 `
  --validation-samples 400 --n-layers 4 --evaluation-batch-size 2 `
  --min-in-range-samples 100 --min-in-range-per-pilot 20 `
  --output-dir D:\EqDeepRxRuns\sensitivity\joint_v1 --resume
```

The summary must contain 18 complete profile/seed records before `scripts/freeze_sensitivity.py` is allowed to select a profile. If Task 2 finds a different confirmed value, add a named profile and rerun the complete profile/seed row rather than silently changing an existing profile's meaning.

- [ ] **Step 2: Confirm the top two profiles at a longer budget.**

Select at most two profiles using the all-seed 2,000-step rule, then run each profile and `formal_baseline` for seeds `2026,2027,2028` at 10,000 steps into `joint_v2_10k`. Keep the exact initialization, validation plan, and four-layer protocol. Require finite state and a per-seed improvement over the formal baseline at 10,000 steps; a profile that wins only at 2,000 steps is not frozen.

- [ ] **Step 3: Stop the search when the finite matrix is exhausted.**

Freeze the lowest pooled BER profile only if it remains best at 10,000 steps, improves all three seeds, has valid coverage, and no competing tested profile is better on the worst-seed metric. If no profile satisfies these conditions, freeze the formal paper-aligned configuration and document the remaining numerical gap as unresolved private implementation detail. Do not add ad-hoc candidates after seeing the 10,000-step result without registering a new matrix and repeating all three seeds.

### Task 4: Produce the pre-training freeze manifest and handoff

**Files:**
- Runtime create: `D:\EqDeepRxRuns\audits\sensitivity_freeze_v2.json`
- Modify: `docs/clean_training_audit.md`
- Modify: `docs/paper_fidelity_audit.md`

- [ ] **Step 1: Run the strict freeze audit.**

```powershell
.\.venv\Scripts\python.exe scripts/freeze_sensitivity.py `
  --root D:\EqDeepRxRuns\sensitivity\confirmation_v2 `
  --joint-summary D:\EqDeepRxRuns\sensitivity\joint_v1\joint_summary.json `
  --output D:\EqDeepRxRuns\audits\sensitivity_freeze_v2.json
```

Require `status=ready_for_long_training` only when all expected records are finite, coverage-valid, provenance-matched, and no formal checkpoint was touched. The manifest must retain `global_optimum_claim=false`, list all rejected values and reasons, list the selected finite-matrix profile, and state that the authors' private random stream is unavailable.

- [ ] **Step 2: Run repository verification without training.**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
git diff --check
```

Expected: all tests pass; no `train.py` or formal evaluator is launched. Update the two audit documents with actual record counts, budgets, selected profile, and unresolved limitations.

- [ ] **Step 3: Ask for explicit long-run approval.**

Report the frozen config fingerprint, candidate/profile metrics, checkpoint-touch status, test count, and the exact remaining caveats. End with a direct question asking whether to start the clean 70,000-step run. Do not start it in the same operation as this plan. Source code, tests, audits, and plans may be published now; runtime checkpoints, plots, and metric JSON remain outside the repository until the clean run and final Figure 6(a) verification.

## Acceptance Checklist

- [ ] Every one of the 39 declared unpublished candidate values has three finite Sionna records at the same 2,000-step budget, or an explicit recorded failure that blocks freezing.
- [ ] `time_mixer_channels=1` and `4` are explicitly present in the confirmation matrix; a 20-step screen cannot discard them.
- [ ] Joint profiles have complete three-seed records and no formal checkpoint writes.
- [ ] The selected profile remains best at the longer confirmation budget, or the formal defaults are retained.
- [ ] Figure 6(a) remains four-layer and excludes DenoiseNN-only.
- [ ] No mathematical global-optimum or bit-identical-paper claim is made.
- [ ] Full tests and the freeze audit pass before asking for approval to run 70,000 steps.
