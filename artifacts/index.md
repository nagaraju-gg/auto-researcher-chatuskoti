# Strong V1 Artifact Index

This directory collects the current public evidence bundle for `Chatuskoti Evals`.

The checked-in bundle is curated from the strongest current saved torch runs. `scripts/run_torch_release_bundle.sh` regenerates the same layout with the stronger `3`-seed configuration.

## Start here

- canonical failure benchmark: [summary.md](strong_v1/canonical_failure/failure_injection/summary.md)
- canonical benchmark figure: [benchmark_figure.svg](strong_v1/canonical_failure/failure_injection/benchmark_figure.svg)
- challenge comparison: [comparison.md](strong_v1/challenge_compare/comparison.md)
- challenge case table: [challenge_cases.md](strong_v1/challenge_compare/challenge_cases.md)
- ablation summary: [summary.md](strong_v1/ablations/summary.md)

## How to read this bundle

1. Read the canonical failure benchmark first.
2. Use the challenge comparison as companion evidence, not as a plain leaderboard.
3. Use the ablation sweep to see which parts of `Vec3` are doing real work.

## Interpretation shortcut

- If binary is higher on the challenge metric but only because it adopted pyrrhic, gamed, or incomparable cases, that is evidence for the value of structured evaluation.
- The canonical failure benchmark is the primary proof artifact.
- The challenge comparison is the “what this looks like inside a loop” artifact.
