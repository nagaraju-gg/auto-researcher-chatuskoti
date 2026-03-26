# Chatuskoti Evals

Most research loops still make one brittle decision: metric up means ship, metric down means reject.

`Chatuskoti Evals` starts from a simpler claim: not every gain is the same kind of gain. A reported improvement can be:

- clean: the benchmark improved and internals stayed healthy
- pyrrhic: the metric improved, but training dynamics degraded
- gamed: the metric improved for suspicious proxy-breaking reasons
- broken: the change damaged the system badly enough that rollback is the right action
- incomparable: the evaluation regime changed, so the before/after comparison is invalid

That is why the project uses a four-state evaluation logic around `truthness`, `coherence`, `comparability`, plus a separate `goodhart_score`, instead of a single binary accept/reject rule.

## Why four states beat binary eval

| Case | What happened | Binary eval | Chatuskoti Evals | Why it matters |
| --- | --- | --- | --- | --- |
| Clean gain | Metric improves and internals stay sane | `adopt` | `adopt` | Good changes should still flow through quickly |
| Pyrrhic gain | Metric rises while instability widens | `adopt` | `hold` | Prevents shipping seductive but unhealthy updates |
| Gamed gain | Metric rises while proxy alignment collapses | `adopt` | `reject` | Catches Goodhart-style false positives |
| Broken result | Metric drops and internals are damaged | `reject` | `rollback` | Separates passive rejection from active damage control |
| Incomparable gain | Eval protocol changed | `adopt` | `reframe` | Stops invalid leaderboard wins from being treated as real progress |

The point is not “more philosophy.” The point is that a single number often throws away the structure that tells you what to do next.

## What this repo proves today

This repo is a benchmark-specific calibrated evaluation framework over `CIFAR-100 + ResNet-18`, with:

- typed intervention proposals
- deterministic `Vec3` scoring
- explicit resolver actions: `adopt`, `reject`, `hold`, `rollback`, `reframe`, `keep_going`
- machine-readable histories, plots, manifests, and markdown reports

It is the evaluation core for a future `Auto Researcher Chatuskoti` loop, not a claim of a fully general autonomous researcher today.

## Strong V1 artifacts

The public-facing evidence bundle is organized under [artifacts/strong_v1](artifacts/strong_v1).

The checked-in bundle is a curated torch-backed benchmark package built from the strongest current saved runs in the repo. The release script regenerates the same bundle in a stronger `3`-seed form on a faster machine.

- canonical failure benchmark: [summary.md](artifacts/strong_v1/canonical_failure/failure_injection/summary.md)
- canonical figure: [benchmark_figure.svg](artifacts/strong_v1/canonical_failure/failure_injection/benchmark_figure.svg)
- challenge comparison: [comparison.md](artifacts/strong_v1/challenge_compare/comparison.md)
- challenge case table: [challenge_cases.md](artifacts/strong_v1/challenge_compare/challenge_cases.md)
- ablation summary: [summary.md](artifacts/strong_v1/ablations/summary.md)
- artifact landing page: [index.md](artifacts/index.md)
- plain-language explanation: [why_four_states.md](docs/why_four_states.md)
- guided walkthrough: [demo.md](docs/demo.md)

## Current headline

The canonical failure benchmark is the main result to lead with.

- binary evaluation would adopt the pyrrhic, gamed, and incomparable benchmark cases
- `Chatuskoti Evals` routes those same cases to `hold`, `reject`, and `reframe`
- the damaged failure case is escalated to `rollback`, not just passively rejected

The benchmark-aware `challenge` comparison is companion evidence:

- binary can post the higher metric by accepting benchmark-aware invalid merges
- `Vec3` preserves structural validity even when that means refusing superficially better numbers

## Quickstart

Environment check:

```bash
.venv/bin/python scripts/check_torch_env.py
```

Regenerate the strongest real benchmark:

```bash
.venv/bin/python -m chatuskoti_evals.cli run-failure-set \
  --backend torch \
  --epochs 10 \
  --seeds 3 \
  --num-workers 0 \
  --output artifacts/strong_v1/canonical_failure
```

Generate the canonical figure:

```bash
python3 scripts/generate_failure_figure.py \
  artifacts/strong_v1/canonical_failure/failure_injection/failure_results.json
```

Run the benchmark-aware open-loop companion:

```bash
.venv/bin/python -m chatuskoti_evals.cli compare \
  --mode challenge \
  --backend torch \
  --epochs 10 \
  --iterations 4 \
  --seeds 3 \
  --num-workers 0 \
  --output artifacts/strong_v1/challenge_compare
```

Run the failure-benchmark ablation sweep:

```bash
.venv/bin/python -m chatuskoti_evals.cli run-ablation \
  --backend torch \
  --epochs 10 \
  --seeds 3 \
  --num-workers 0 \
  --output artifacts/strong_v1/ablations
```

Or regenerate the whole release bundle:

```bash
bash scripts/run_torch_release_bundle.sh
```

## What gets written

Each controller or benchmark bundle emits:

- `history.jsonl`
- `seed_metrics.json`
- `summary.md`
- SVG plots
- aggregate summaries
- `manifest.json`

The stronger public bundles also emit:

- challenge case tables
- benchmark figures
- ablation summaries
- top-level artifact index pages

## Package and CLI

The Python package is now `chatuskoti_evals`.

- module entrypoint: `python -m chatuskoti_evals.cli`
- project name: `chatuskoti-evals`

The most useful commands are:

- `compare`
- `run-loop`
- `run-failure-set`
- `run-ablation`

## Scope and claims

This repo should be presented as:

- benchmark-specific
- calibrated to `CIFAR-100 + ResNet-18`
- evaluation-first
- reproducible and interpretable

This repo should not yet be presented as:

- a general automated researcher
- universally calibrated across domains
- proof that `Vec3` always beats binary controllers on any benchmark

## Future AR integration

This project is designed to plug into a broader research loop:

1. an external AR system proposes an intervention
2. the benchmark runner executes it and emits `RunMetrics`
3. `Chatuskoti Evals` scores the result as `Vec3 + goodhart_score`
4. the resolver returns `adopt`, `reject`, `hold`, `rollback`, `reframe`, or `keep_going`
5. history and wisdom feed back into the next proposal

That future system is the right place for `Auto Researcher Chatuskoti`. This repo is the eval layer that makes such a loop safer and more legible.

## Repo structure

- [chatuskoti_evals/benchmark.py](chatuskoti_evals/benchmark.py)
- [chatuskoti_evals/scoring.py](chatuskoti_evals/scoring.py)
- [chatuskoti_evals/resolver.py](chatuskoti_evals/resolver.py)
- [chatuskoti_evals/proposals.py](chatuskoti_evals/proposals.py)
- [chatuskoti_evals/reporting.py](chatuskoti_evals/reporting.py)
- [docs/why_four_states.md](docs/why_four_states.md)
- [docs/demo.md](docs/demo.md)
- [docs/release_checklist.md](docs/release_checklist.md)

## Test suite

```bash
python3 -m unittest discover -s tests -v
```
