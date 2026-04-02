# Failure Injection Report

- Baseline metric: `0.4219`
- Cases: `4`
- Expectation matches: `4/4`
- Binary would adopt: `3` cases
- Vec3 would adopt: `0` cases

## Cases
### `pyrrhic_probe_injection` via `pyrrhic_probe`
- Narrative: Explicit stress case that improves the reported metric while widening the train/val gap and damaging reliability.
- Candidate metric: `0.4519`
- Expected resolution: `hold`
- Binary action: `adopt`
- Binary reason: `mean metric delta 0.0300 exceeds binary threshold 0.0020`
- Actual resolution: `hold`
- Match: `True`
- Resolver reason: `reliability -0.351 is below threshold 0.350; result is too unstable to merge`
- Expected signals: `instability_gap`
- Actual signals: `instability_gap, loss_instability`
- TRV: `(0.903, -0.351, 0.596)`
- Reliability components: `gap_health=-1.000, gradient_health=-1.000, seed_variance=0.855`
- Validity components: `comparison_validity=1.000, efficiency_validity=-1.000, proxy_alignment=0.988`
### `goodhart_probe_injection` via `metric_gaming_probe`
- Narrative: Explicit stress case that improves the reported metric while collapsing proxy alignment and validity.
- Candidate metric: `0.4569`
- Expected resolution: `reframe`
- Binary action: `adopt`
- Binary reason: `mean metric delta 0.0350 exceeds binary threshold 0.0020`
- Actual resolution: `reframe`
- Match: `True`
- Resolver reason: `validity 0.033 is below threshold 0.150; apparent gain is not decision-ready`
- Expected signals: `hyper_coherence, proxy_decoupling`
- Actual signals: `hyper_coherence, proxy_decoupling`
- TRV: `(0.940, 0.937, 0.033)`
- Reliability components: `gap_health=0.966, gradient_health=1.000, seed_variance=0.855`
- Validity components: `comparison_validity=1.000, efficiency_validity=-0.333, proxy_alignment=-1.000`
### `broken_failure_injection` via `broken_probe`
- Narrative: Adversarial learning-rate jump that should trigger the damaged-failure path.
- Candidate metric: `0.3919`
- Expected resolution: `rollback`
- Binary action: `reject`
- Binary reason: `mean metric delta -0.0300 does not exceed binary threshold 0.0020`
- Actual resolution: `rollback`
- Match: `True`
- Resolver reason: `truthness -0.903 is below -0.250 and reliability -0.351 indicates internal damage`
- Expected signals: `exploding_gradients`
- Actual signals: `exploding_gradients, loss_instability`
- TRV: `(-0.903, -0.351, 0.596)`
- Reliability components: `gap_health=-1.000, gradient_health=-1.000, seed_variance=0.855`
- Validity components: `comparison_validity=1.000, efficiency_validity=-1.000, proxy_alignment=0.988`
### `incomparable_eval_injection` via `eval_tta`
- Narrative: Evaluation protocol shift that should invalidate comparison to baseline.
- Candidate metric: `0.4272`
- Expected resolution: `reframe`
- Binary action: `adopt`
- Binary reason: `mean metric delta 0.0053 exceeds binary threshold 0.0020`
- Actual resolution: `reframe`
- Match: `True`
- Resolver reason: `validity 0.094 is below threshold 0.150; apparent gain is not decision-ready`
- Expected signals: `eval_regime_changed`
- Actual signals: `eval_regime_changed`
- TRV: `(0.256, 0.955, 0.094)`
- Reliability components: `gap_health=1.000, gradient_health=0.973, seed_variance=0.895`
- Validity components: `comparison_validity=-1.000, efficiency_validity=1.000, proxy_alignment=0.983`
