# Paper Fidelity and Sensitivity Audit

This audit separates values stated in *EqDeepRx: Learning a Scalable and Interference Mitigating MIMO Receiver* from implementation choices that the paper does not make executable.

## Source-backed public configuration

The source used for the audit is `D:\EqDeepRxRuns\eq_deeprx_rev.tex`. The current manifest is generated with:

```powershell
.\.venv\Scripts\python.exe scripts/paper_fidelity_audit.py `
  --paper-source D:\EqDeepRxRuns\eq_deeprx_rev.tex `
  --output D:\EqDeepRxRuns\audits\paper_fidelity.json
```

The manifest currently reports 30 public rows, all matched against the source and runtime configuration. These include `F=192`, `S=14`, `N_R=16`, primary `N_T=4`, `B=8`, `E=2`, the shared training coverage `N_T=2--4`, 64-QAM, native UMa training, the **validation** delay-spread range `10--1100 ns`, the SNR/speed/INR distributions (including the published lognormal mean and spread), one TX antenna per UE, DMRS spacing and count, the Table I widths/subsampling, four DetectorNN sections, 24-subcarrier INCM, `alpha=1e-4`, batch 112, learning rate `4.4e-3`, `lambda=1e-5`, and `alpha_mVCL=1e-5`. The exact CDL validation sampling plumbing is not numerically specified by the paper and remains an explicit implementation choice. The paper's separate `10--100 ns` reduced-delay-spread BLER experiment is not part of Figure 6(a).

The public architecture and equations are covered by executable tests in `tests/test_paper_fidelity.py`, `tests/test_signal.py`, `tests/test_sionna_system.py`, and `tests/test_model.py`. Those tests also lock the Sionna Kronecker pilot contract: staggered every-fourth-subcarrier masks, reserved pilot symbols, and QPSK sequence reuse across selected DMRS symbols. The primary Figure 6(a) evaluator uses four layers; training still samples the shared model over 2, 3, and 4 layers.

## Choices not numerically specified by the paper

The manifest records these values without calling them paper facts: FFT/CP, the TimeMixer subset `C_s`, residual projection bias and other exact convolution bias/initialization details, LAMB beta/epsilon/weight decay and bias correction, AMP boundary, mVCL attachment, symbol-loss reduction, UMa/CDL delay-spread plumbing, fixed-vs-sampled CDL delay spread, CIR discretization/normalization, pathloss and shadow-fading switches, interferer timing window, random stream, and exact SINR bin edges. The residual skip projection is explicitly configurable and defaults to `bias=False`, matching the current graph; this is an exploratory variable, not a paper claim.

The formal defaults remain the existing implementation values: `C_s=2`, FFT/CP `256/18`, sampled CDL delay spread `U(10,1100) ns`, sinc CIR with normalization, pathloss and shadow fading disabled, random-symbol interferer offset, LAMB `(0.9,0.999,1e-6,0)`, bfloat16 convolution boundaries with float32 loss boundaries, mVCL on all four Detector states, and a summed masked symbol norm. These are choices, not claims about private author code. The historical schema-5 Sionna matrix is complete as screening-plus-confirmation evidence: 17 variables, 75 finite records, screening seeds `2026/2027` at 20 steps, and confirmation seed `2028` at 2,000 steps; no record touched a formal checkpoint. Because those budgets differ, its provisional labels do not establish a same-budget three-seed optimum. The next `confirmation_v2` pass enumerates the complete 39-value matrix and runs all three seeds at the same budget; only its freeze audit can select an exploratory value. The primary Figure 6(a) evaluator contains only the two EqDeepRx curves and the three specified baseline/reference curves.

## Sensitivity protocol

`scripts/run_unpublished_sensitivity.py` changes exactly one choice per job. Initialization, training seed stream, layer count (four for the Figure 6(a) protocol), pilot coverage, validation sample IDs, displayed SINR centers, and protocol fields are fixed. The baseline is the value encoded by the formal `paper_config()`, independent of candidate-list order. The historical screening pass used two short seeds and one longer confirmation seed; the release supervisor's equal-budget pass uses seeds 2026, 2027, and 2028 at the same explicit budget. Results are written below `D:\EqDeepRxRuns\sensitivity` and never to a formal checkpoint.

Candidate values are:

| Variable | Values |
| --- | --- |
| TimeMixer channels | `1, 2, 4` |
| CDL validation delay spread mode | `uniform_10_1100ns`, `fixed_300ns`, `fixed_100ns`, `fixed_1000ns` |
| UMa delay spread | `native`, `uniform_10_1100ns` |
| CIR discretization | `sinc`, `nearest` |
| CIR normalization | `true`, `false` |
| Pathloss / shadow fading | `false`, `true` independently |
| Interferer timing | `random_symbol`, `zero`, `random_sample` |
| LAMB beta1 / beta2 / epsilon / weight decay / bias correction | one-at-a-time alternatives in the script |
| AMP boundary | `bfloat16`, `float32` |
| mVCL attachment | `all`, `final`, `none` |
| Symbol-loss reduction | `sum`, `mean_active` |
| Residual skip projection bias | `false`, `true` |

A candidate is eligible for the short screening label only if its same-sample BER improves over the baseline for both 20-step screening seeds, the Sionna validation has the required in-range coverage, and all states remain finite. The equal-budget confirmation supervisor then enumerates every value in the complete candidate matrix and runs seeds 2026, 2027, and 2028 with the same explicit `--confirmation-steps` value of at least 2,000; the earlier `confirmation_v1` records remain historical screening evidence only. `scripts/run_sensitivity_confirmation_supervisor.ps1` defaults to `-SelectionMode all`, passes the equal budget to both worker step arguments, and supports a dry-run audit of that job list. Fast-backend records with no realized-SINR coverage are retained for debugging but are never eligible for promotion. The harness writes an `incomplete` summary while any expected JSON is missing and can resume only schema-5 records whose candidate/configuration, independent system seed, and validation-plan fingerprint match. The validation fingerprint also binds layer count, evaluation batch size, channel model, and speed range, preventing diagnostic protocols from being mixed with the formal comparison. An improvement that changes a public paper value is exploratory and cannot replace the formal paper-aligned configuration. No numerical claim of matching the paper is made until a clean 70k run and a digitizable reference comparison exist.

## Current freeze gate

`D:\EqDeepRxRuns\audits\sensitivity_freeze_prejoint.json` is the machine-readable
summary of the completed one-variable matrix. It records the formal paper-aligned
profile separately from the best confirmed exploratory profile:

```text
formal profile: C_s=2, sampled CDL U(10,1100 ns), LAMB bias correction on,
                and all other published/default choices
confirmed exploratory candidates: CDL fixed 100 ns, CDL fixed 300 ns,
                                  LAMB bias correction off
```

The two fixed-delay candidates are both provisionally confirmed by the
20-step/2,000-step screening protocol, with `fixed_100ns` having the lower
pooled three-seed recorded metric in this finite matrix. Selecting it and
turning off LAMB bias correction together is an additional joint choice, not a
mathematical global-optimum proof. On 2026-09-28 the first equal-budget joint record
completed (`formal_baseline`, seed `2026`, Sionna, four layers, 2,000 steps,
pooled BER `0.209386304607957`, 161/400 in-range samples); the user then paused
the matrix to release the GPU. The partial
`D:\EqDeepRxRuns\sensitivity\joint_v1\joint_summary.json` correctly remains
`status=incomplete`, so no exploratory value is frozen and the formal paper
profile remains the only launch-safe choice. The next session can resume the
six-profile, three-seed command with `--resume`; the full-candidate confirmation
pass is a separate prerequisite for claiming that every unpublished choice was
tested. The finite experiments cannot
establish the authors' private random stream or guarantee numerical identity
with the paper curve; the final 70k result must retain that limitation.

## Figure reference

The repository contains prior reproduction assets, but no machine-readable author Figure 6(a) coordinates. `scripts/digitize_paper_figure6a.py` therefore records asset hashes and reports `unavailable` unless explicit coordinates are supplied; it does not infer coordinates from an unrelated reproduction image.
