# BINARY Loop Report

- Initial baseline metric: `0.4186`
- Final accepted metric (controller's own eval): `0.4600`
- Final canonical benchmark metric: `0.4600`
- Final baseline metric after adoptions: `0.4600`
- Iterations: `4`
- Fired signals seen: `eval_regime_changed, exploding_gradients, instability_gap, loss_instability`

## Iterations
### Iteration 1: `pyrrhic_probe`
- Action: `adopt`
- Why: `mean metric delta 0.0300 exceeds binary threshold 0.0020`
- Vec3: `(0.905, -0.780, 0.750)`
- Goodhart score: `0.000`
- Signals: `instability_gap, loss_instability`
### Iteration 2: `metric_gaming_probe`
- Action: `adopt`
- Why: `mean metric delta 0.0050 exceeds binary threshold 0.0020`
- Vec3: `(0.245, 0.650, 0.750)`
- Goodhart score: `0.000`
- Signals: `none`
### Iteration 3: `eval_tta`
- Action: `adopt`
- Why: `mean metric delta 0.0064 exceeds binary threshold 0.0020`
- Vec3: `(0.310, 0.650, -1.000)`
- Goodhart score: `0.000`
- Signals: `eval_regime_changed`
### Iteration 4: `adamw`
- Action: `reject`
- Why: `mean metric delta -0.0038 does not exceed binary threshold 0.0020`
- Vec3: `(-0.188, -0.920, 0.750)`
- Goodhart score: `0.000`
- Signals: `exploding_gradients`
