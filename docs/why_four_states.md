# Why Four States

Most evaluation systems make one coarse distinction: improvement or no improvement.

That is too lossy for research loops. A model update can improve the reported metric while still being the wrong thing to merge. `Chatuskoti Evals` separates those cases so the control decision matches the kind of result we actually observed.

## The five practical cases

| Case | Short version | Resolver action |
| --- | --- | --- |
| Clean gain | Better metric, healthy internals, valid comparison | `adopt` |
| Pyrrhic gain | Better metric, worse training dynamics | `hold` |
| Gamed gain | Better metric, suspicious proxy behavior | `reject` |
| Broken result | Worse metric, obvious damage | `rollback` |
| Incomparable gain | Better metric under a changed eval regime | `reframe` |

The important shift is this:

- binary evaluation asks “did the number go up?”
- four-state evaluation asks “what kind of outcome is this, and what action fits it?”

## How the scoring works

`Chatuskoti Evals` scores each outcome along:

- `truthness`: did the anchored benchmark metric really improve?
- `coherence`: did internals remain healthy?
- `comparability`: is this result even comparable to the baseline?
- `goodhart_score`: did the metric move for suspicious reasons?

That extra structure turns evaluation into control logic. A pyrrhic gain should not be treated like a clean gain, and an incomparable gain should not even be treated as a valid before/after result.

## Why this matters for research loops

In real loops, many bad decisions do not look bad at first:

- the metric gets a little better
- the training process gets less stable
- the proxy metrics decouple
- the evaluation quietly changes

If the controller only sees the top-line number, it will merge exactly the kinds of results that later become regressions, misleading wins, or wasted search effort.

That is what this repo is trying to fix.
