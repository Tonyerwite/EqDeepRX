# EqDeepRx Reproduction

This directory contains the final PyTorch/Sionna implementation of the
64-QAM EqDeepRx receiver through uncoded BER. The implementation includes the
paper-aligned signal model, RZF/LMMSE equalization, DenoiseNN, shared
DetectorNN, DemapperNN, loss, LAMB schedule, Sionna channel generation,
Figure 6(a) evaluation, and resumable training checkpoints.

The final paper-aligned configuration uses the TimeMixer subset `C_s=2`. No
additional algorithmic parameter is applied. The public configuration covers
192 subcarriers, 14 OFDM symbols, 16 RX antennas, 2/3/4 training layers,
1/2-DMRS training cases, effective batch 112, initial learning rate 4.4e-3,
and 70,000 optimizer steps.

## Install

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-standard.txt
```

## macOS preflight

Run the preflight before a clean run. On the tested Apple M3 Pro with 18 GiB
unified memory, the approved settings are `microbatch=28`, generation batch
`2`, one Sionna worker, and 12 CPU threads. The measured estimate is about
15.3 days for 70,000 steps.

```bash
.venv/bin/python scripts/preflight.py \
  --device mps --microbatch-size 28 --generation-batch-size 2 \
  --sionna-workers 1 --sionna-cpu-threads 12 \
  --output /tmp/preflight_mps_final.json
```

## Train

Start a new clean run with a single checkpoint containing DenoiseNN,
DetectorNN, DemapperNN, one shared LAMB optimizer, history, RNG state, and
the run configuration:

```bash
.venv/bin/python scripts/train.py --confirm-full-run \
  --steps 70000 --batch-size 112 --microbatch-size 28 \
  --generation-batch-size 2 --sionna-workers 1 --sionna-cpu-threads 12 \
  --device mps --seed 2026 \
  --preflight-report /tmp/preflight_mps_final.json \
  --save-every 500 --output /tmp/eqdeeprx_runs/eqdeeprx_clean70k.pt
```

Resume an interrupted run with the same arguments and checkpoint path:

```bash
.venv/bin/python scripts/train.py --confirm-full-run \
  --steps 70000 --batch-size 112 --microbatch-size 28 \
  --generation-batch-size 2 --sionna-workers 1 --sionna-cpu-threads 12 \
  --device mps --seed 2026 \
  --preflight-report /tmp/preflight_mps_final.json \
  --resume /tmp/eqdeeprx_runs/eqdeeprx_clean70k.pt \
  --save-every 500 --output /tmp/eqdeeprx_runs/eqdeeprx_clean70k.pt
```

The trainer rejects mismatched or incomplete checkpoints. Every saved state
requires matching global steps and history lengths, finite model and optimizer
states, equal optimizer counters for all trainable parameters, saved RNG
states, and `checkpoint_status=complete`.

## Figure 6(a)

Evaluate only a completed 70,000-step checkpoint. This produces the two
EqDeepRx curves and the three conventional reference curves; no DenoiseNN-only
curve is used.

```bash
.venv/bin/python scripts/evaluate_uncoded_ber.py \
  --checkpoint /tmp/eqdeeprx_runs/eqdeeprx_clean70k.pt \
  --validation-samples 32000 --evaluation-batch-size 2 --n-layers 4 \
  --seed 2026 --device mps --resume \
  --output-dir /tmp/eqdeeprx_runs/figure6a_clean70k
```

The paper does not publish every executable detail or the authors' private
random stream, so exact bit-for-bit equality with the private reference cannot
be inferred. The code and training path are complete and protocol-aligned.
