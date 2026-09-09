# EqDeepRx Reproduction

This project reproduces the EqDeepRx receiver from **EqDeepRx: Learning a Scalable and Interference Mitigating MIMO Receiver** (Honkala, Korpi, Raninen, Huttunen) up to uncoded bit error rate (BER). It is a PyTorch implementation that reuses the signal-processing boundary established by the supplied DeepRx OFDM reference while implementing the EqDeepRx hybrid architecture.

The deliverable intentionally stops before LDPC decoding. It reports hard decisions from the learned bit logits and conventional baselines as uncoded BER; it does not claim the paper's post-decoding BLER or spectral-efficiency results.

## What is implemented

- DMRS-based raw channel estimation, interpolation, complex covariance estimation with shrinkage, parallel RZF and interference-aware LMMSE equalizers, and unit-gain scaling.
- DenoiseNN on each RX/TX pilot channel pair.
- Shared-weight per-layer DetectorNN with coordinate maps and full/1:8 residual paths.
- Shared per-layer DemapperNN with eight output logits; active modulation bits are masked.
- Direct uncoded QAM targets, equation (13) weighted BCE plus symbol loss, LAMB, and linear learning-rate decay.
- Deterministic tiny smoke generation and a five-curve uncoded-BER evaluator.

## Paper defaults

The default `paper_config()` uses the EqDeepRx paper's Table I/II settings: 30 kHz SCS, 192 subcarriers, 14 OFDM symbols, 16 RX antennas, 2-4 MIMO layers, UMa training, 64-QAM, SNR 0-45 dB, 1 or 2 DMRS symbols, RZF `alpha=1e-4`, 24-subcarrier INCM coherence bandwidth, batch size 112, LAMB learning rate `4.4e-3`, approximately 70,000 iterations, and symbol-loss weight `1e-5`. The network has 118,740 trainable parameters in this implementation, close to the paper's 116k count excluding equalizers.

## Setup and checks

Run from this directory. A Python 3.9 environment with PyTorch 2.8 and CUDA 12.8 was used for the local verification.

```powershell
py -3 -m pip install -r requirements.txt
py -3 -m pytest -q
py -3 scripts/preflight.py
```

The preflight checks importability, CUDA visibility, model size, input-memory estimate, and does not start training. Use the tiny CPU check when validating a new machine:

```powershell
py -3 scripts/preflight.py --tiny --device cpu
```

## Safe smoke run

The tiny path uses 16 active subcarriers, 2 RX antennas, 2 layers, and a one-step update. It is the required first run before any paper-scale job:

```powershell
py -3 scripts/train.py --tiny --steps 1 --batch-size 1 --device cpu --output outputs/smoke.pt
py -3 scripts/evaluate_uncoded_ber.py --tiny --snr-points 0,6 --samples-per-point 1 --device cpu --output-dir outputs/smoke_eval
```

## Paper-scale training (not started by default)

Run preflight first. The command below is the full training entry point, but it is guarded by an explicit confirmation flag and was not launched as part of this delivery:

```powershell
py -3 scripts/train.py --steps 70000 --batch-size 112 --n-layers 4 --pilot-count 1 --device cuda --seed 2026 --output checkpoints/eqdeeprx.pt --confirm-full-run
```

The guard is the `scripts/train.py --confirm-full-run` confirmation; without it, paper-scale training is refused.

The expected memory is dominated by the 112-sample, 16-RX, 192-subcarrier feature maps and activations. Do not run the paper-scale job concurrently with BER Monte Carlo on an 8 GB GPU. Checkpoint files are ignored by Git.

## Uncoded BER evaluation

```powershell
py -3 scripts/evaluate_uncoded_ber.py --checkpoint checkpoints/eqdeeprx.pt --samples-per-point 100 --n-layers 4 --snr-points 0,3,6,9,12,15,18,21 --output-dir outputs/uncoded_ber
```

The evaluator writes `uncoded_ber_metrics.json` and `uncoded_ber.png`. It compares EqDeepRx with practical LMMSE, known-channel LMMSE, and one/two-pilot configurations. It does not call an LDPC decoder.

## Audit trail

- Paper-to-code and DeepRx inheritance audit: `docs/paper_audit.md`.
- External open-source search evidence: `docs/open_source_search.md`.
- Local preflight evidence before full training: `docs/preflight_report.md`.
