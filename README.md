# Chatuskoti Eval Framework

Four-state evaluation framework for research-loop decisions on `CIFAR-100 + ResNet-18`. Instead of asking only “did the metric go up?”, it asks whether a candidate run is:

- actually better
- stable enough to trust
- valid enough to compare and merge

The name comes from Gautama's [Chatuskoti](https://en.wikipedia.org/wiki/Catu%E1%B9%A3ko%E1%B9%ADi): a four-way logic lens that is useful when binary true/false decisions throw away important structure. In this repo that idea is implemented as a practical three-axis evaluator, not as philosophy-first labeling.

The basic motivation is:

| Question | Correct mode | Why |
| --- | --- | --- |
| "Paris is the capital of France" | True | Plain factual statement |
| "Paris is the capital of Germany" | False | Plain factual error |
| "Light is a wave" | Both | Useful in one frame, incomplete in another |
| "What is the color of number 7?" | Neither | Ill-posed category error |

Most benchmarks are good at the first two and weak on the latter two. The same failure shows up in research loops: not every apparent "gain" is the same kind of gain.

The practical intuition is simple:

| Outcome type | What binary sees | What the evaluator should notice |
| --- | --- | --- |
| Clean gain | metric up | safe to merge |
| Pyrrhic gain | metric up | run is unstable |
| Metric-gamed gain | metric up | improvement is not decision-ready |
| Broken result | metric down | run is damaged, not just weak |
| Incomparable gain | metric up | comparison itself is invalid |

## Headline result

On the current torch-backed V1.1 benchmark bundle under [artifacts/strong_v1_1_torch](artifacts/strong_v1_1_torch):

- the canonical failure benchmark matches `4/4` expected cases
- binary evaluation would adopt `3/4` benchmark-aware bad cases
- `Vec3` routes those cases to `hold`, `reframe`, and `rollback`
- in challenge mode, binary reaches the higher metric by merging three cases the benchmark says should not be merged

That is the core claim of this repo: structured evaluation can preserve benchmark validity even when a metric-only controller posts a better number.

## How the evaluator works

The evaluator decomposes the decision into three axes:

| Axis | What it asks | Why this decomposition matters for four-state logic |
| --- | --- | --- |
| `truthness` (`T`) | Did the benchmark really improve? | Preserves the ordinary true/false question instead of throwing it away. |
| `reliability` (`R`) | Did the result remain stable and reproducible enough to trust? | Separates ordinary improvement from pyrrhic cases where the top-line number rises but the run is unstable. |
| `validity` (`V`) | Is the apparent gain meaningful rather than gamed or comparison-invalid? | Separates real progress from metric-gaming and evaluation-regime failures that should be treated as neither cleanly true nor false. |

Those axes drive the controller actions that binary evaluation collapses together:

| Case | Binary eval | `Vec3` | Why |
| --- | --- | --- | --- |
| Clean gain | `adopt` | `adopt` | Better metric, healthy run, valid comparison |
| Pyrrhic gain | `adopt` | `hold` | Metric improves while internals destabilize |
| Metric-gamed gain | `adopt` | `reframe` | Top-line gain is not decision-ready progress |
| Broken result | `reject` | `rollback` | Damage is explicit enough to justify revert semantics |
| Incomparable gain | `adopt` | `reframe` | Evaluation regime changed, so the comparison is invalid |

In V1.1, metric-gaming evidence is not a separate public axis. It is exposed as part of `validity`, alongside explicit sub-signals for proxy decoupling, comparison breaks, and inefficient gains.

## What this repo proves today

This repo is a benchmark-specific calibrated evaluation framework over `CIFAR-100 + ResNet-18`, with:

- typed intervention proposals
- deterministic `Vec3` scoring
- inspectable `T/R/V` subcomponents in reports and logs
- explicit resolver actions: `adopt`, `reject`, `hold`, `rollback`, `reframe`, `keep_going`
- machine-readable histories, plots, manifests, and markdown reports
- append-only offline learning logs at `logs/runs.jsonl` (gitignored)

The strongest current local evidence package shows that a metric-only controller can merge benchmark-aware bad cases that the structured evaluator treats differently for specific, inspectable reasons.

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
  --output artifacts/strong_v1_1_torch/canonical_failure
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
  --output artifacts/strong_v1_1_torch/challenge_compare
```

Run the failure-benchmark ablation sweep:

```bash
.venv/bin/python -m chatuskoti_evals.cli run-ablation \
  --backend torch \
  --epochs 10 \
  --seeds 3 \
  --num-workers 0 \
  --output artifacts/strong_v1_1_torch/ablations
```

Or regenerate the whole release bundle:

```bash
bash scripts/run_torch_release_bundle.sh
```

## Current evidence bundle

The strongest current torch-backed bundle is under [artifacts/strong_v1_1_torch](artifacts/strong_v1_1_torch).

- canonical failure benchmark: [summary.md](artifacts/strong_v1_1_torch/canonical_failure/failure_injection/summary.md)
  This is the main result. It matches `4/4` expected benchmark cases and cleanly separates pyrrhic, metric-gamed, broken, and incomparable outcomes.
- canonical figure: [benchmark_figure.svg](artifacts/strong_v1_1_torch/canonical_failure/failure_injection/benchmark_figure.svg)
  Fastest visual summary of the torch benchmark.
- challenge comparison: [comparison.md](artifacts/strong_v1_1_torch/challenge_compare/comparison.md)
  Companion result showing binary achieves the higher metric by adopting three benchmark-aware bad cases.
- challenge case table: [challenge_cases.md](artifacts/strong_v1_1_torch/challenge_compare/challenge_cases.md)
  Exact divergence table for `pyrrhic_probe`, `metric_gaming_probe`, and `eval_tta`.
- ablation summary: [summary.md](artifacts/strong_v1_1_torch/ablations/summary.md)
  Compact evidence that both `reliability` and `validity` matter: removing either drops benchmark matches from `4/4` to `2/4`.
- artifact landing page: [index.md](artifacts/index.md)
- plain-language explanation: [why_four_states.md](docs/why_four_states.md)
- guided walkthrough: [demo.md](docs/demo.md)

## Headline result

The canonical failure benchmark is the main result to lead with.

- binary evaluation would adopt the pyrrhic, metric-gamed, and incomparable benchmark cases
- `Chatuskoti Eval Framework` routes those same cases to `hold`, `reframe`, and `reframe`
- the damaged failure case is escalated to `rollback`, not just passively rejected

The benchmark-aware `challenge` comparison is companion evidence:

- binary posts the higher metric by accepting benchmark-aware invalid merges
- `Vec3` preserves structural validity even when that means refusing superficially better numbers

## What gets written

Each controller or benchmark bundle emits:

- `history.jsonl`
- `seed_metrics.json`
- `summary.md`
- SVG plots
- aggregate summaries
- `manifest.json`

The evaluator also appends compact run records to `logs/runs.jsonl` for future offline calibration. That file is intentionally gitignored so normal experimentation does not bloat the repo.

## Why V1.1 is not just a rename

V1.1 changes more than labels:

- `T` stays anchored to the benchmark metric, so the evaluator remains grounded in the repo's original purpose.
- `R` is explicitly statistical and operational: spread, gradient instability, and train/val gap damage.
- `V` unifies anti-Goodhart and comparison-validity logic, but the underlying causes stay visible through subcomponents and fired signals.
- every scored run now leaves behind an offline-learning record instead of throwing those features away.

The technical claim is therefore narrower but stronger: this is a cleaner, more measurable benchmark-specific evaluator, not a leap to a learned general system.

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

## Future: full Chatuskoti eval benchmark

This repo is not yet a full general-purpose Chatuskoti eval benchmark in the strict sense.

It would cross that line when it has:

1. locked task definitions with strict output schemas
2. a larger high-quality dataset, including adversarial `both` and `neither` cases
3. deterministic scoring and formalized rubrics
4. published baseline model comparisons
5. clone-and-run reproducibility with stable headline numbers

That is a natural future for this repo. The current version is a benchmark-specific calibrated evaluation framework that moves in that direction.

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
