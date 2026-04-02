# Next Steps

This repo is already organized around the `strong_v1_1_torch` public bundle. The next step is to regenerate or extend that bundle on a machine with `torch` and `torchvision` installed.

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

## 3. Run the V1.1 bundle

Use the prepared wrapper:

```bash
bash scripts/run_torch_release_bundle.sh
```

That script generates:

- `artifacts/strong_v1_1_torch/canonical_failure`
- `artifacts/strong_v1_1_torch/challenge_compare`
- `artifacts/strong_v1_1_torch/ablations`

## 4. Or run the main pieces directly

If you want to run pieces one at a time:

```bash
.venv/bin/python -m chatuskoti_evals.cli run-failure-set \
  --backend torch \
  --epochs 10 \
  --seeds 3 \
  --num-workers 0 \
  --output artifacts/strong_v1_1_torch/canonical_failure
```

```bash
.venv/bin/python -m chatuskoti_evals.cli compare \
  --mode challenge \
  --backend torch \
  --epochs 10 \
  --iterations 4 \
  --seeds 3 \
  --num-workers 0 \
  --output artifacts/strong_v1_1_torch/challenge_compare
```

```bash
.venv/bin/python -m chatuskoti_evals.cli run-ablation \
  --backend torch \
  --epochs 10 \
  --seeds 3 \
  --num-workers 0 \
  --output artifacts/strong_v1_1_torch/ablations
```

Expected outputs:

- `artifacts/strong_v1_1_torch/canonical_failure/failure_injection/summary.md`
- `artifacts/strong_v1_1_torch/challenge_compare/comparison.md`
- `artifacts/strong_v1_1_torch/ablations/summary.md`
- JSON manifests, per-seed metrics, and SVG charts

If the smoke test succeeds and the machine is CUDA-capable, then scale up to `--epochs 30`.

## 5. Check the evidence gate

Do not call it publish-ready yet unless all are true:

- the canonical failure benchmark matches all intended branches
- binary still adopts pyrrhic, Goodhart-style, and incomparable benchmark cases that `Vec3` blocks
- the challenge comparison still shows structural invalid merges being accepted by binary
- the ablation summary weakens in the expected directions
- outputs are reproducible across reruns within a reasonable tolerance band

## 6. Tighten before publishing

After the first run, inspect and likely refine:

- threshold calibration in the detector config
- failure probes and challenge-mode behavior, if the benchmark behavior drifts
- paper wording so claims match real evidence rather than simulator behavior
- package/release metadata so the package version, docs, and artifact bundle name stay aligned

## 7. Minimum publish package

For a credible first public drop, include:

- the repo
- exact run command
- one saved `strong_v1_1_torch` bundle
- the canonical failure figure
- the challenge comparison table
- the ablation summary
- one short limitations section that says this is a benchmark-specific calibrated evaluation framework
