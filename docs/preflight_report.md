# Preflight Report

The following checks were run before any paper-scale training:

```text
py -3 -m pytest -q
17 passed in 4.73s

py -3 scripts/preflight.py --device cpu
{
  "torch": "2.8.0+cu128",
  "cuda_available": true,
  "device": "cpu",
  "paper_steps": 70000,
  "model_parameters": 118740,
  "estimated_received_input_mib": 36.75,
  "full_training_started": false
}

Default forward check: `[1, 2, 8, 192, 14]`, all values finite, 118,740 parameters.

py -3 scripts/preflight.py --tiny
tiny_forward_shape = [1, 2, 4, 16, 14]
tiny_forward_finite = true

py -3 scripts/train.py --tiny --steps 1 --batch-size 1 --device cpu
steps = 1, finite loss and BER, atomic checkpoint created

py -3 scripts/evaluate_uncoded_ber.py --tiny --snr-points 0,6 --samples-per-point 1 --device cpu
five finite curves written to outputs/smoke_eval/uncoded_ber_metrics.json
```

No cache, 70k-step job, LDPC decoder, or external publication action was started. The full-scale command is guarded by `--confirm-full-run` and remains a user decision.
