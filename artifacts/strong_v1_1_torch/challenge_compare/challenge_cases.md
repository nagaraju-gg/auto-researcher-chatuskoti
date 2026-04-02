# Challenge Case Table

| Action | Binary | Vec3 | Signals | Why It Matters |
| --- | --- | --- | --- | --- |
| `pyrrhic_probe` | `adopt` | `hold` | `instability_gap, loss_instability` | Pyrrhic gain: metric improved while internals destabilized. |
| `metric_gaming_probe` | `adopt` | `reframe` | `hyper_coherence, proxy_decoupling` | Metric-gaming risk: validity collapsed despite a top-line gain. |
| `eval_tta` | `adopt` | `reframe` | `eval_regime_changed` | Invalid comparison: the apparent gain is not decision-ready. |
| `adamw` | `reject` | `rollback` | `exploding_gradients, loss_instability` | Damaged run: controller should actively revert. |
