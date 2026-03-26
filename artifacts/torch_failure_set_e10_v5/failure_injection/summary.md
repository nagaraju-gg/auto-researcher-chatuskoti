# Failure Injection Report

- Baseline metric: `0.4186`
- Cases: `4`
- Expectation matches: `4/4`
- Binary would adopt: `3` cases
- Vec3 would adopt: `0` cases

## Cases
### `pyrrhic_probe_injection` via `pyrrhic_probe`
- Narrative: Explicit stress case that improves the reported metric while widening the train/val gap and damaging coherence.
- Candidate metric: `0.4486`
- Expected resolution: `hold`
- Binary action: `adopt`
- Binary reason: `mean metric delta 0.0300 exceeds binary threshold 0.0020`
- Actual resolution: `hold`
- Match: `True`
- Resolver reason: `truthness 0.905 is positive but coherence -0.780 is below -0.350`
- Expected signals: `instability_gap`
- Actual signals: `instability_gap, loss_instability`
- Vec3: `(0.905, -0.780, 0.750)`
- Goodhart score: `0.000`
### `goodhart_probe_injection` via `metric_gaming_probe`
- Narrative: Explicit stress case that improves the reported metric while collapsing proxy alignment and triggering Goodhart detection.
- Candidate metric: `0.4536`
- Expected resolution: `reject`
- Binary action: `adopt`
- Binary reason: `mean metric delta 0.0350 exceeds binary threshold 0.0020`
- Actual resolution: `reject`
- Match: `True`
- Resolver reason: `goodhart_score 0.840 exceeds threshold 0.650 while truthness is positive`
- Expected signals: `hyper_coherence, proxy_decoupling`
- Actual signals: `hyper_coherence, proxy_decoupling`
- Vec3: `(0.941, 0.530, 0.750)`
- Goodhart score: `0.840`
### `broken_failure_injection` via `broken_probe`
- Narrative: Adversarial learning-rate jump that should trigger the damaged-failure path.
- Candidate metric: `0.3886`
- Expected resolution: `rollback`
- Binary action: `reject`
- Binary reason: `mean metric delta -0.0300 does not exceed binary threshold 0.0020`
- Actual resolution: `rollback`
- Match: `True`
- Resolver reason: `truthness -0.905 is below -0.250 and coherence -0.920 indicates damage`
- Expected signals: `exploding_gradients`
- Actual signals: `exploding_gradients, loss_instability`
- Vec3: `(-0.905, -0.920, 0.750)`
- Goodhart score: `0.000`
### `incomparable_eval_injection` via `eval_tta`
- Narrative: Evaluation protocol shift that should invalidate comparison to baseline.
- Candidate metric: `0.4250`
- Expected resolution: `reframe`
- Binary action: `adopt`
- Binary reason: `mean metric delta 0.0064 exceeds binary threshold 0.0020`
- Actual resolution: `reframe`
- Match: `True`
- Resolver reason: `comparability -1.000 is below -0.800; baseline comparison is not valid`
- Expected signals: `eval_regime_changed`
- Actual signals: `eval_regime_changed`
- Vec3: `(0.310, 0.650, -1.000)`
- Goodhart score: `0.000`
