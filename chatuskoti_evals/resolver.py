from __future__ import annotations

from chatuskoti_evals.config import DetectorConfig
from chatuskoti_evals.models import Resolution, RunMetrics, RunScore


def resolve_vec3(run_score: RunScore, cfg: DetectorConfig) -> Resolution:
    t = run_score.mean.truthness
    c = run_score.mean.coherence
    k = run_score.mean.comparability

    if run_score.mag < cfg.min_magnitude:
        return Resolution("keep_going", f"signal magnitude {run_score.mag:.3f} is below minimum {cfg.min_magnitude:.3f}")
    if cfg.enable_spread_gate and run_score.spread > cfg.max_spread:
        return Resolution("keep_going", f"seed spread {run_score.spread:.3f} exceeds maximum {cfg.max_spread:.3f}")
    if cfg.enable_goodhart and run_score.goodhart_score >= cfg.goodhart_threshold and t > cfg.adopt_truth_threshold:
        return Resolution(
            "reject",
            f"goodhart_score {run_score.goodhart_score:.3f} exceeds threshold {cfg.goodhart_threshold:.3f} while truthness is positive",
        )
    if cfg.enable_coherence and t > cfg.adopt_truth_threshold and c <= -cfg.incoherence_threshold:
        return Resolution(
            "hold",
            f"truthness {t:.3f} is positive but coherence {c:.3f} is below -{cfg.incoherence_threshold:.3f}",
        )
    if cfg.enable_coherence and t < cfg.reject_truth_threshold and c <= -cfg.incoherence_threshold:
        return Resolution(
            "rollback",
            f"truthness {t:.3f} is below {cfg.reject_truth_threshold:.3f} and coherence {c:.3f} indicates damage",
        )
    if cfg.enable_comparability and k <= -cfg.comparability_threshold:
        return Resolution(
            "reframe",
            f"comparability {k:.3f} is below -{cfg.comparability_threshold:.3f}; baseline comparison is not valid",
        )
    if t > cfg.adopt_truth_threshold:
        return Resolution("adopt", f"truthness {t:.3f} exceeds adopt threshold {cfg.adopt_truth_threshold:.3f}")
    return Resolution("reject", f"truthness {t:.3f} does not exceed adopt threshold {cfg.adopt_truth_threshold:.3f}")


def resolve_binary(candidate_metrics: list[RunMetrics], baseline_metrics: RunMetrics, cfg: DetectorConfig) -> Resolution:
    mean_metric = sum(item.primary_metric for item in candidate_metrics) / len(candidate_metrics)
    metric_delta = mean_metric - baseline_metrics.primary_metric
    if metric_delta > cfg.binary_metric_threshold:
        return Resolution("adopt", f"mean metric delta {metric_delta:.4f} exceeds binary threshold {cfg.binary_metric_threshold:.4f}")
    return Resolution("reject", f"mean metric delta {metric_delta:.4f} does not exceed binary threshold {cfg.binary_metric_threshold:.4f}")
