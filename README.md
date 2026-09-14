# EqDeepRx Reproduction

This repository reproduces the primary EqDeepRx receiver from **EqDeepRx: Learning a Scalable and Interference Mitigating MIMO Receiver** through uncoded BER. It intentionally excludes LDPC decoding, BLER, spectral efficiency, DenoiseNN-only, monolithic, and other ablation lines.

The standard path trains one 64-QAM model online on UMa data while randomly covering 2/3/4 MIMO layers, one/two DMRS symbols, and batches with/without one interfering UE. DetectorNN and DemapperNN share weights across layers, so the resulting checkpoint is used for every covered layer count and DMRS configuration without retraining.

## Implemented Paper Path

- Sionna 2.1 TR 38.901 UMa/UMi/CDL-C/CDL-D time-domain channels, OFDM modulation/demodulation, AWGN, and an independently faded interfering UE with random timing offset.
- Orthogonal staggered DMRS, rank-one raw channel estimates, learned per-RX/TX-pair DenoiseNN, and linear interpolation to the full resource grid.
- Complex OAS shrinkage of the 24-subcarrier INCM, parallel unit-gain LMMSE and RZF (`alpha=1e-4`) equalizers.
- Shared per-layer DetectorNN with coordinate maps and full/1:8 residual blocks, followed by shared per-layer DemapperNN with eight LLR outputs.
- Paper loss: bit-0-positive weighted BCE, four DetectorNN symbol losses with `lambda=1e-5`, and mVCL mean/variance regularization with `alpha=1e-5`.
- LAMB at `4.4e-3`, linear decay to zero, 70,000 optimizer steps, and effective batch 112 (about 7.84 million online samples).
- Paper Figure 6(a) uncoded-BER protocol: CDL-C, 10-15 m/s, one interferer, realized-SINR binning, and 32,000 validation slots total.

The primary network has **115,456 trainable parameters**, excluding deterministic equalizers, consistent with the paper's rounded 116k count.

## Environment

The delivered environment was verified with Python 3.12, PyTorch 2.9.1+cu128, and Sionna 2.1.0. Run commands from this directory:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-standard.txt
.\.venv\Scripts\python.exe -m pytest -q
```

## Smoke Checks

The tiny backend is only for fast CPU regression checks; standard training and Figure 6(a) evaluation require the Sionna time-domain backend.

```powershell
.\.venv\Scripts\python.exe scripts/preflight.py --tiny --device cpu
.\.venv\Scripts\python.exe scripts/train.py --tiny --steps 1 --batch-size 1 --microbatch-size 1 --generation-batch-size 1 --device cpu --output outputs/smoke.pt
.\.venv\Scripts\python.exe scripts/evaluate_uncoded_ber.py --tiny --sinr-points 0,6 --samples-per-point 1 --device cpu --output-dir outputs/smoke_eval
```

## Full Preflight

Run this immediately before long training. It warms up and measures all 12 combinations of 2/3/4 layers, one/two DMRS, and interference absent/present, then writes the configuration gate consumed by the trainer.

```powershell
.\.venv\Scripts\python.exe scripts/preflight.py --device cuda --microbatch-size 28 --generation-batch-size 2 --output outputs/preflight_standard.json
```

On the verified NVIDIA GeForce RTX 5060 Laptop GPU (8 GiB), this configuration covered all 12 cases with finite loss/gradients, used 4007 MiB peak allocated CUDA memory, measured 12.298 samples/s, and estimated **7.378 days** for 70,000 steps. The optimization keeps the paper algorithm, effective batch, training length, network, loss, and channel distribution unchanged; only CUDA AMP (bfloat16 for the learned convolution stacks, float32 at complex/loss boundaries), generation batching, model microbatching, and duplicate CIR computation were optimized.

## Full Training

The following is the single formal training command. Do not add `--n-layers` or `--pilot-count`: leaving both unset is what samples every paper configuration into one shared checkpoint.

```powershell
.\.venv\Scripts\python.exe scripts/train.py --confirm-full-run --steps 70000 --batch-size 112 --microbatch-size 28 --generation-batch-size 2 --device cuda --seed 2026 --preflight-report outputs/preflight_standard.json --output checkpoints/eqdeeprx.pt
```

Resume an interrupted run without changing its configuration:

```powershell
.\.venv\Scripts\python.exe scripts/train.py --confirm-full-run --steps 70000 --batch-size 112 --microbatch-size 28 --generation-batch-size 2 --device cuda --seed 2026 --preflight-report outputs/preflight_standard.json --resume checkpoints/eqdeeprx.pt --output checkpoints/eqdeeprx.pt
```

The full run is never started automatically. The trainer rejects paper-scale runs unless CUDA, exact steps/batch size, configuration fingerprint, and approved preflight batch sizes all match.

## Figure 6(a) Uncoded BER

After training, generate the requested uncoded-BER result. The 32,000 samples are the total across both DMRS configurations, matching the paper wording, and are binned by realized SINR rather than requested SNR. The paper-figure reference uses three MIMO layers and SINR points `-5,-3,-1,1,3,5,7,9,11,13` dB; `--n-layers` remains available for the other scalable layer counts.

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_uncoded_ber.py --checkpoint checkpoints/eqdeeprx.pt --validation-samples 32000 --evaluation-batch-size 2 --device cuda --resume --output-dir outputs/figure6a
```

The evaluator writes `figure6a_metrics.json`, resumable progress, and `figure6a_uncoded_ber.png`. It reports EqDeepRx and the conventional comparison curves needed to interpret the reproduced EqDeepRx line; it does not run a decoder.

## Reproduction Boundary

All architecture, formulas, and numeric settings stated in the public paper are mapped in `docs/paper_audit.md`. The paper does not publish its original source code, exact FFT/CP selection, TimeMixer subset width, LAMB beta/epsilon/weight decay, seed, exact SINR bin edges, or exact mVCL attachment points. Those necessary choices are recorded rather than presented as author-private facts. Consequently, this is a paper-aligned independent reproduction, not a claim of bit-identical output to unavailable private code.

See `docs/preflight_report.md` for the final readiness evidence and `docs/open_source_search.md` for the public-code search result.
