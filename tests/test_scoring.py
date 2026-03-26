from __future__ import annotations

import unittest

from chatuskoti_evals.config import DetectorConfig
from chatuskoti_evals.models import RunMetrics
from chatuskoti_evals.resolver import resolve_vec3
from chatuskoti_evals.scoring import score_run_metrics


def make_metrics(
    *,
    run_id: str,
    primary_metric: float,
    train_loss: float,
    val_loss: float,
    train_val_gap: float,
    grad_norm_mean: float,
    grad_norm_std: float,
    eval_hash: str = "baseline",
    objective_family: str = "cross_entropy",
    proxy_corr: float = 0.86,
) -> RunMetrics:
    return RunMetrics(
        run_id=run_id,
        seed=0,
        primary_metric=primary_metric,
        train_loss=train_loss,
        val_loss=val_loss,
        train_val_gap=train_val_gap,
        grad_norm_mean=grad_norm_mean,
        grad_norm_std=grad_norm_std,
        weight_distance=0.2,
        param_count=11_200_000,
        eval_hash=eval_hash,
        model_family="resnet18",
        objective_family=objective_family,
        proxy_metrics={"proxy_metric_corr": proxy_corr, "calibration": max(0.0, proxy_corr - 0.05)},
        detector_inputs={},
    )


class ScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = DetectorConfig()
        self.baseline = make_metrics(
            run_id="baseline",
            primary_metric=0.636,
            train_loss=1.45,
            val_loss=1.63,
            train_val_gap=0.18,
            grad_norm_mean=2.1,
            grad_norm_std=0.11,
        )

    def score(self, metric: RunMetrics):
        run_score, _ = score_run_metrics([metric, metric, metric], self.baseline, self.cfg)
        return run_score

    def test_clean_win_routes_to_adopt(self) -> None:
        metric = make_metrics(
            run_id="clean-win",
            primary_metric=0.671,
            train_loss=1.36,
            val_loss=1.49,
            train_val_gap=0.13,
            grad_norm_mean=1.9,
            grad_norm_std=0.10,
        )
        run_score = self.score(metric)
        self.assertGreater(run_score.mean.truthness, 0.25)
        self.assertGreater(run_score.mean.coherence, 0.0)
        resolution = resolve_vec3(run_score, self.cfg)
        self.assertEqual(resolution.action, "adopt")
        self.assertIn("exceeds adopt threshold", resolution.reason)

    def test_pyrrhic_win_routes_to_hold(self) -> None:
        metric = make_metrics(
            run_id="pyrrhic",
            primary_metric=0.668,
            train_loss=1.30,
            val_loss=1.72,
            train_val_gap=0.42,
            grad_norm_mean=2.0,
            grad_norm_std=0.10,
        )
        run_score = self.score(metric)
        self.assertIn("instability_gap", run_score.fired_signals)
        self.assertEqual(resolve_vec3(run_score, self.cfg).action, "hold")

    def test_goodhart_win_routes_to_reject(self) -> None:
        metric = make_metrics(
            run_id="goodhart",
            primary_metric=0.669,
            train_loss=1.32,
            val_loss=1.54,
            train_val_gap=0.22,
            grad_norm_mean=1.7,
            grad_norm_std=0.02,
            proxy_corr=0.42,
        )
        run_score = self.score(metric)
        self.assertGreaterEqual(run_score.goodhart_score, self.cfg.goodhart_threshold)
        self.assertEqual(resolve_vec3(run_score, self.cfg).action, "reject")

    def test_incomparable_routes_to_reframe(self) -> None:
        metric = make_metrics(
            run_id="incomparable",
            primary_metric=0.649,
            train_loss=1.39,
            val_loss=1.57,
            train_val_gap=0.18,
            grad_norm_mean=1.95,
            grad_norm_std=0.10,
            eval_hash="changed",
            objective_family="focal_loss",
        )
        run_score = self.score(metric)
        self.assertLess(run_score.mean.comparability, 0.0)
        self.assertEqual(resolve_vec3(run_score, self.cfg).action, "reframe")

    def test_noisy_lucky_seed_routes_to_keep_going(self) -> None:
        metrics = [
            make_metrics(
                run_id=f"noisy-{index}",
                primary_metric=value,
                train_loss=1.38,
                val_loss=1.55,
                train_val_gap=0.17,
                grad_norm_mean=1.95,
                grad_norm_std=0.09,
            )
            for index, value in enumerate([0.70, 0.61, 0.67])
        ]
        run_score, _ = score_run_metrics(metrics, self.baseline, self.cfg)
        self.assertGreater(run_score.spread, self.cfg.max_spread)
        self.assertEqual(resolve_vec3(run_score, self.cfg).action, "keep_going")

    def test_positive_metric_with_isolated_grad_mean_spike_can_still_adopt(self) -> None:
        metric = make_metrics(
            run_id="adamw-like",
            primary_metric=0.670,
            train_loss=1.31,
            val_loss=1.45,
            train_val_gap=0.14,
            grad_norm_mean=11.8,
            grad_norm_std=0.14,
            proxy_corr=0.88,
        )
        run_score = self.score(metric)
        self.assertNotIn("exploding_gradients", run_score.fired_signals)
        self.assertGreater(run_score.mean.coherence, 0.0)
        self.assertEqual(resolve_vec3(run_score, self.cfg).action, "adopt")
