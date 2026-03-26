# Demo Walkthrough

This repo is easiest to understand in three stops.

## 1. Canonical failure benchmark

Start here:

- [summary.md](../artifacts/strong_v1/canonical_failure/failure_injection/summary.md)
- [benchmark_figure.svg](../artifacts/strong_v1/canonical_failure/failure_injection/benchmark_figure.svg)

This is the strongest artifact in the repo. It deliberately constructs four benchmark-aware stress cases:

- pyrrhic gain
- Goodhart-style gain
- broken/damaged result
- incomparable eval shift

The intended read is simple:

- binary eval would merge several of these cases
- `Chatuskoti Evals` routes each one differently because they are different kinds of outcomes

## 2. Challenge comparison

Then read:

- [comparison.md](../artifacts/strong_v1/challenge_compare/comparison.md)
- [challenge_cases.md](../artifacts/strong_v1/challenge_compare/challenge_cases.md)

This is the open-loop-style companion result. It is not meant to be read as a plain leaderboard. The important question is whether a higher binary metric came from accepting cases the benchmark says should not be merged.

That is why the challenge report explicitly lists:

- which cases binary adopted
- what `Vec3` did instead
- which signals fired

## 3. Ablation sweep

Finally read:

- [summary.md](../artifacts/strong_v1/ablations/summary.md)

This answers the question: do the extra `Vec3` axes actually matter?

Expected pattern:

- remove `coherence` and pyrrhic handling gets worse
- remove `goodhart_score` and metric-gaming handling gets worse
- remove `comparability` and eval-shift handling gets worse

## Limitations

This is a strong v1, not a universal claim.

- one benchmark: `CIFAR-100 + ResNet-18`
- typed interventions, not open-ended code synthesis
- benchmark-specific calibration
- evaluation layer, not a full autonomous research system

That narrowness is deliberate. The goal is to make the core decision logic legible and reproducible before expanding breadth.
