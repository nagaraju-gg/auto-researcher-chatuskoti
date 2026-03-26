# Challenge Case Table

| Action | Binary | Vec3 | Signals | Why It Matters |
| --- | --- | --- | --- | --- |
| `pyrrhic_probe` | `adopt` | `hold` | `instability_gap, loss_instability` | Pyrrhic gain: metric improved while internals destabilized. |
| `metric_gaming_probe` | `adopt` | `reject` | `hyper_coherence, proxy_decoupling` | Goodhart-style gain: metric improved for suspicious reasons. |
| `eval_tta` | `adopt` | `reframe` | `eval_regime_changed` | Incomparable gain: evaluation regime changed. |
| `adamw` | `reject` | `reject` | `exploding_gradients, loss_instability` | Rejected as a non-improving or unstable change. |
