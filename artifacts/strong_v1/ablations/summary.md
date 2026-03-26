# Failure Benchmark Ablations

This report re-resolves the saved canonical failure benchmark under ablated detector settings.

| Ablation | Matched | Case outcomes |
| --- | --- | --- |
| `full` | `4/4` | pyrrhic_probe_injection -> hold, goodhart_probe_injection -> reject, broken_failure_injection -> rollback, incomparable_eval_injection -> reframe |
| `no_coherence` | `2/4` | pyrrhic_probe_injection -> adopt, goodhart_probe_injection -> reject, broken_failure_injection -> reject, incomparable_eval_injection -> reframe |
| `no_comparability` | `3/4` | pyrrhic_probe_injection -> hold, goodhart_probe_injection -> reject, broken_failure_injection -> rollback, incomparable_eval_injection -> adopt |
| `no_goodhart` | `3/4` | pyrrhic_probe_injection -> hold, goodhart_probe_injection -> adopt, broken_failure_injection -> rollback, incomparable_eval_injection -> reframe |
| `no_wisdom` | `4/4` | pyrrhic_probe_injection -> hold, goodhart_probe_injection -> reject, broken_failure_injection -> rollback, incomparable_eval_injection -> reframe |
| `no_spread_gate` | `4/4` | pyrrhic_probe_injection -> hold, goodhart_probe_injection -> reject, broken_failure_injection -> rollback, incomparable_eval_injection -> reframe |
