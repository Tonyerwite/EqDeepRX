# Final Preflight Report

The formal CUDA preflight was run with the delivered Sionna/PyTorch environment and did not start the 70,000-step job.

```powershell
.\.venv\Scripts\python.exe scripts/preflight.py --device cuda --microbatch-size 28 --generation-batch-size 2 --output outputs/preflight_standard.json
```

Recorded result:

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
Measured throughput            12.038 samples/s
Estimated optimizer step       9.304 s
Estimated 70000-step runtime   7.538 days
long_training_ready            true
```

The estimate is based on warmed runs of all 12 supported configurations, not cold startup. Generation batch 2 avoids the Sionna memory spike; model microbatch 28 reduces accumulation overhead while retaining the exact effective batch 112 and full-batch mVCL statistics. CUDA AMP restores float32 at learned-module boundaries and checkpoints the GradScaler with a verified initial scale of 1.0.

The full trainer additionally checks the preflight configuration fingerprint and approved batch sizes before accepting `--confirm-full-run`. Checkpoints are atomic and include model, optimizer, scaler, schedule position, history, and configuration for exact resume validation.
