# Binary vs Vec3 Comparison

- Mode: `challenge`
- Baseline metric: `0.4186`
- Vec3 final canonical benchmark metric: `0.4186`
- Binary final canonical benchmark metric: `0.4600`

## Interpretation

- Verdict: `binary` is higher on canonical benchmark metric for this challenge run, but that metric must be read together with benchmark-aware invalid merges.
- Vec3 final canonical metric: `0.4186`
- Binary final canonical metric: `0.4600`
- Vec3 non-adopt actions: `4`
- Binary rejects: `1`
- Vec3 fired signals: `eval_regime_changed, exploding_gradients, hyper_coherence, instability_gap, loss_instability, proxy_decoupling`
- Binary fired signals: `eval_regime_changed, exploding_gradients, instability_gap, loss_instability`

## Readout

- This challenge run is designed to test whether the controller distinguishes benchmark-aware bad merges from clean improvements.
- The main signal here is not final metric alone; it is whether each controller adopts or blocks pyrrhic, Goodhart-style, and incomparable cases.
- Treat this as companion evidence to the canonical failure benchmark rather than as a plain unconstrained leaderboard comparison.

## Challenge-specific readout

- Binary adopted `3` benchmark-aware cases that `Vec3` did not adopt.
- `pyrrhic_probe`: binary `adopt` vs Vec3 `hold` (instability_gap, loss_instability)
- `metric_gaming_probe`: binary `adopt` vs Vec3 `reject` (hyper_coherence, proxy_decoupling)
- `eval_tta`: binary `adopt` vs Vec3 `reframe` (eval_regime_changed)
- In `challenge` mode, final metric should be read together with structural validity, not as the only success criterion.
- A higher binary metric here can reflect merges that the benchmark is explicitly designed to classify as pyrrhic, Goodhart-style, or incomparable.
