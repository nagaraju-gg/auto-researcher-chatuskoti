from __future__ import annotations

import math
from statistics import mean, stdev

from catuskoti_ar.config import DetectorConfig
from catuskoti_ar.models import RunMetrics, RunScore, SeedScore, Vec3


def score_run_metrics(
    candidate_metrics: list[RunMetrics],
    baseline_metrics: RunMetrics,
    cfg: DetectorConfig,
) -> tuple[RunScore, list[SeedScore]]:
    if not candidate_metrics:
        raise ValueError("candidate_metrics must not be empty")

    per_seed_scores = [score_single_seed(run, baseline_metrics, cfg) for run in candidate_metrics]

    truthnesses = [seed.vec3.truthness for seed in per_seed_scores]
    coherences = [seed.vec3.coherence for seed in per_seed_scores]
    comparabilities = [seed.vec3.comparability for seed in per_seed_scores]
    goodhart_scores = [seed.goodhart_score for seed in per_seed_scores]

    mean_vec = Vec3(
        truthness=round(mean(truthnesses), 5),
        coherence=round(mean(coherences), 5),
        comparability=round(mean(comparabilities), 5),
    )
    std_vec = Vec3(
        truthness=round(stdev(truthnesses), 5) if len(truthnesses) > 1 else 0.0,
        coherence=round(stdev(coherences), 5) if len(coherences) > 1 else 0.0,
        comparability=round(stdev(comparabilities), 5) if len(comparabilities) > 1 else 0.0,
    )
    mag = round(
        math.sqrt(mean_vec.truthness**2 + mean_vec.coherence**2 + mean_vec.comparability**2),
        5,
    )
    spread = round(
        math.sqrt(std_vec.truthness**2 + std_vec.coherence**2 + std_vec.comparability**2),
        5,
    )

    fired_signals: list[str] = []
    raw_detectors: dict[str, float] = {}
    for seed_score in per_seed_scores:
        for signal in seed_score.fired_signals:
            if signal not in fired_signals:
                fired_signals.append(signal)
        for key, value in seed_score.raw_detectors.items():
            if isinstance(value, (int, float)):
                raw_detectors.setdefault(key, 0.0)
                raw_detectors[key] += float(value)
    raw_detectors = {key: round(value / len(per_seed_scores), 5) for key, value in raw_detectors.items()}

    return RunScore(
        mean=mean_vec,
        std=std_vec,
        mag=mag,
        spread=spread,
        goodhart_score=round(mean(goodhart_scores), 5),
        fired_signals=fired_signals,
        raw_detectors=raw_detectors,
    ), per_seed_scores


def score_single_seed(
    candidate: RunMetrics,
    baseline: RunMetrics,
    cfg: DetectorConfig,
) -> SeedScore:
    fired_signals: list[str] = []
    raw_detectors: dict[str, float | bool | str] = {}

    metric_delta = candidate.primary_metric - baseline.primary_metric
    truthness = safe_tanh(metric_delta / cfg.truth_delta_scale)

    negative_penalty = 0.0
    gap_ratio = ratio(candidate.train_val_gap, baseline.train_val_gap)
    grad_mean_ratio = ratio(candidate.grad_norm_mean, baseline.grad_norm_mean)
    grad_std_ratio = ratio(candidate.grad_norm_std, baseline.grad_norm_std)
    proxy_corr_delta = candidate.proxy_metrics["proxy_metric_corr"] - baseline.proxy_metrics["proxy_metric_corr"]

    raw_detectors.update(
        {
            "metric_delta": metric_delta,
            "gap_ratio": gap_ratio,
            "grad_mean_ratio": grad_mean_ratio,
            "grad_std_ratio": grad_std_ratio,
            "proxy_corr_delta": proxy_corr_delta,
        }
    )

    if not all(map(math.isfinite, [candidate.train_loss, candidate.val_loss, candidate.primary_metric])):
        fired_signals.append("nan_loss")
        negative_penalty = max(negative_penalty, 1.0)
    if gap_ratio > cfg.gap_multiplier and metric_delta > -0.01:
        fired_signals.append("instability_gap")
        negative_penalty = max(negative_penalty, 0.78)
    corroborated_grad_damage = (
        grad_mean_ratio > cfg.grad_mean_multiplier
        and (
            grad_std_ratio > cfg.grad_std_multiplier
            or gap_ratio > cfg.gap_multiplier
            or metric_delta <= 0.0
        )
    )
    if corroborated_grad_damage:
        fired_signals.append("exploding_gradients")
        negative_penalty = max(negative_penalty, 0.92)
    if grad_std_ratio > cfg.grad_std_multiplier:
        fired_signals.append("loss_instability")
        negative_penalty = max(negative_penalty, 0.68)

    if negative_penalty > 0:
        coherence = -negative_penalty
    else:
        coherence_bonus = 0.45
        if candidate.train_val_gap < baseline.train_val_gap:
            coherence_bonus += 0.12
        if candidate.grad_norm_mean <= baseline.grad_norm_mean:
            coherence_bonus += 0.08
        coherence = min(coherence_bonus, 0.78)

    comparability = 0.75
    if candidate.eval_hash != baseline.eval_hash:
        fired_signals.append("eval_regime_changed")
        comparability = -1.0
    elif candidate.model_family != baseline.model_family:
        fired_signals.append("model_family_changed")
        comparability = -1.0
    elif candidate.objective_family != baseline.objective_family:
        fired_signals.append("objective_changed")
        comparability = -0.9

    goodhart_score = 0.0
    if metric_delta >= cfg.improvement_for_goodhart and grad_std_ratio < cfg.hypercoherence_ratio:
        fired_signals.append("hyper_coherence")
        goodhart_score = max(goodhart_score, 0.72)
    if metric_delta >= cfg.improvement_for_goodhart and proxy_corr_delta < -cfg.proxy_corr_drop:
        fired_signals.append("proxy_decoupling")
        goodhart_score = max(goodhart_score, 0.84)

    raw_detectors["goodhart_score"] = goodhart_score

    return SeedScore(
        vec3=Vec3(round(truthness, 5), round(coherence, 5), round(comparability, 5)),
        goodhart_score=round(goodhart_score, 5),
        fired_signals=fired_signals,
        raw_detectors=raw_detectors,
    )


def ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-8:
        return 0.0
    return numerator / denominator


def safe_tanh(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return math.tanh(value)
