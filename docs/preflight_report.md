# CUDA Preflight Evidence

The repository's historical preflight record (from the earlier published
commit) is retained below for provenance, but it is not evidence for the
current working tree. The current audit has not started the 70,000-step job.

## Current working-tree measurement

The latest uncontended measurement was written to
`D:\EqDeepRxRuns\audits\preflight_after_gpu_clear.json` on 2026-09-25 after
the other GPU jobs had exited. It reported:

```text
GPU                            NVIDIA GeForce RTX 5060 Laptop GPU
Backend                        sionna_tr38901_time_domain
Model parameters               115456
Configuration coverage         12/12 (2/3/4 layers x 1/2 DMRS x interference off/on)
Loss / gradients               finite / finite
Measured optimizer step        9.933 s
Estimated 70000-step runtime   8.047 days
long_training_ready            false
```

This is a measured runtime estimate, not a training result. The estimate is
0.047 days over the repository's advisory eight-day convenience threshold;
the user has explicitly waived that convenience threshold for this task. It
must remain visible in status reports, but it is not by itself a launch
blocker. The 12/12 configuration coverage, finite loss/gradients, and 4.0 GiB
of measured CUDA headroom are valid evidence that the implementation can
execute the standard path. Numerical-safety and checkpoint-contract gates
remain mandatory.

## Historical measurement

The values below came from the earlier published commit and are kept only to
show why they cannot be reused without a fresh run:

```powershell
.\.venv\Scripts\python.exe scripts/preflight.py --device cuda --microbatch-size 28 --generation-batch-size 2 --output outputs/preflight_standard.json
```

Recorded result after the AMP overflow fix:

```text
Python                         3.12
PyTorch                        2.9.1+cu128
Sionna                         2.1.0
GPU                            NVIDIA GeForce RTX 5060 Laptop GPU (8150.6 MiB)
Backend                        sionna_tr38901_time_domain
Model parameters               115456
Configuration coverage         12/12 (2/3/4 layers x 1/2 DMRS x interference off/on)
Loss / gradients               finite / finite
Approved effective batch       112
Approved model microbatch      28
Approved generation batch      2
Peak CUDA allocation           4007.0 MiB
CUDA headroom                  4143.5 MiB
Measured throughput            12.298 samples/s
Estimated optimizer step       9.107 s
Estimated 70000-step runtime   7.378 days
long_training_ready            true
```

The estimate is based on warmed runs of all 12 supported configurations, not cold startup. Generation batch 2 avoids the Sionna memory spike; model microbatch 28 reduces accumulation overhead while retaining the exact effective batch 112 and full-batch mVCL statistics. CUDA AMP uses bfloat16 inside the DenoiseNN and DetectorNN convolution stacks, which preserves the float32 exponent range and avoids the float16 overflow observed at the failed run; DenoiseNN outputs, DetectorNN states, the DemapperNN, complex operations, and losses remain float32. The GradScaler is checkpointed with a verified initial scale of 1.0.

The full trainer additionally checks the preflight configuration fingerprint and approved batch sizes before accepting `--confirm-full-run`. Checkpoints are atomic and include model, optimizer, scaler, schedule position, history, and configuration for exact resume validation.
