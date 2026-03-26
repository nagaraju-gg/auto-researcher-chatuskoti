# Next Steps

This repo is prepared for the first real benchmark run, but that run must happen on a machine with `torch` and `torchvision` installed.

## 1. Prepare the machine

Run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-torch.txt
.venv/bin/python scripts/check_torch_env.py
```

Success looks like:

- `torch` imports cleanly
- `torchvision` imports cleanly
- at least one GPU backend is available, ideally CUDA

## 2. Run the smoke test first

Before the real comparison, validate the backend with the smallest useful run:

```bash
bash scripts/run_torch_smoke.sh
```

This uses:

- `1` epoch
- `1` iteration
- `1` seed
- `num_workers=0`

If that succeeds, move to the real comparison.

## 3. Run the first real comparison

Use the prepared wrapper:

```bash
bash scripts/run_torch_compare.sh
```

That wrapper is intentionally a lighter real-comparison profile for first contact:

- `10` epochs, not `30`
- `num_workers=0`, which is safer on Apple Silicon / MPS
- progress is printed every epoch so you can tell the run is alive
- it prefers `.venv/bin/python` automatically, so it avoids shell/path confusion
 - it still runs `4` iterations and `3` seeds, so it is much heavier than the smoke test

Or run directly:

```bash
.venv/bin/python -m catuskoti_ar.cli compare \
  --backend torch \
  --epochs 10 \
  --num-workers 0 \
  --output artifacts/torch_compare
```

Expected outputs:

- `artifacts/torch_compare/vec3/summary.md`
- `artifacts/torch_compare/binary/summary.md`
- `artifacts/torch_compare/comparison.md`
- JSONL history, per-seed metrics, and SVG charts

If the smoke test succeeds and the machine is CUDA-capable, then scale up to `--epochs 30`.

## 4. Check the first evidence gate

Do not call it publish-ready yet unless all are true:

- `Vec3` beats `binary` on the canonical benchmark metric
- at least one Goodhart or pyrrhic case is clearly rejected by `Vec3` and accepted by `binary`
- the failure-injection set still behaves as expected under the real backend
- outputs are reproducible across reruns within a reasonable tolerance band

## 5. Tighten before publishing

After the first run, inspect and likely refine:

- threshold calibration in the detector config
- torch-side failure-injection probes and calibration thresholds, if the benchmark behavior drifts
- paper wording so claims match real evidence rather than simulator behavior

## 6. Minimum publish package

For a credible first public drop, include:

- the repo
- exact run command
- one saved canonical run
- three core figures:
  - pyrrhic win
  - Goodhart case
  - recovery via Vec3 history
- one short limitations section that says this is a benchmark-specific calibrated prototype
