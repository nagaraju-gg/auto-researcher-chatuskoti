from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from catuskoti_ar.benchmark import create_benchmark_adapter
from catuskoti_ar.config import ExperimentConfig, LoopConfig
from catuskoti_ar.actions import ACTION_INDEX
from catuskoti_ar.models import BaselineRecord, FailureCaseResult, HistoryEntry, RunMetrics, to_jsonable
from catuskoti_ar.progress import RunProgressContext, RunProgressTracker
from catuskoti_ar.proposals import ProposalEngine
from catuskoti_ar.reporting import ReportGenerator
from catuskoti_ar.resolver import resolve_binary, resolve_vec3
from catuskoti_ar.scenarios import get_failure_injection_set
from catuskoti_ar.scoring import score_run_metrics
from catuskoti_ar.wisdom import WisdomStore


@dataclass(frozen=True)
class LoopResult:
    controller: str
    initial_baseline: BaselineRecord
    final_baseline: BaselineRecord
    history: list[HistoryEntry]
    accepted_metric: float
    raw_accepted_metric: float
    output_dir: Path


def run_single_loop(
    cfg: ExperimentConfig,
    loop_cfg: LoopConfig,
    output_root: Path,
    *,
    progress: RunProgressTracker | None = None,
) -> LoopResult:
    adapter = create_benchmark_adapter(cfg)
    proposal_engine = ProposalEngine()
    report_generator = ReportGenerator(output_root)
    controller_root = output_root / loop_cfg.controller
    controller_root.mkdir(parents=True, exist_ok=True)

    wisdom_path = controller_root / "wisdom_store.json"
    wisdom = WisdomStore.load(wisdom_path)
    seeds = list(range(loop_cfg.n_seeds))
    progress = progress or RunProgressTracker(total_runs=_loop_run_count(loop_cfg))

    initial_baseline = adapter.record_baseline(
        seeds,
        progress=progress,
        progress_context=RunProgressContext(controller=loop_cfg.controller, phase="baseline"),
    )
    current_baseline = initial_baseline
    history: list[HistoryEntry] = []
    per_iteration_metrics: list[list[RunMetrics]] = []
    raw_accepted_metric = current_baseline.metrics.primary_metric

    for iteration in range(1, loop_cfg.max_iterations + 1):
        action = proposal_engine.propose(loop_cfg.controller, history, wisdom, mode=loop_cfg.mode)
        candidate_metrics, candidate_state = adapter.execute(
            action,
            seeds,
            progress=progress,
            progress_context=RunProgressContext(
                controller=loop_cfg.controller,
                phase="iteration",
                iteration=iteration,
                action_name=action.name,
            ),
        )
        compared_baseline_id = current_baseline.baseline_id
        run_score, _ = score_run_metrics(candidate_metrics, current_baseline.metrics, cfg.detector)

        if loop_cfg.controller == "vec3":
            resolution = resolve_vec3(run_score, cfg.detector)
        else:
            resolution = resolve_binary(candidate_metrics, current_baseline.metrics, cfg.detector)

        if resolution.action == "adopt":
            adapter.adopt(candidate_state)
            progress.add_runs(len(seeds))
            current_baseline = adapter.record_baseline(
                seeds,
                progress=progress,
                progress_context=RunProgressContext(
                    controller=loop_cfg.controller,
                    phase="adopted_baseline",
                    iteration=iteration,
                    action_name=action.name,
                ),
            )
            raw_accepted_metric = current_baseline.metrics.primary_metric

        entry = HistoryEntry(
            iteration=iteration,
            timestamp=datetime.now(UTC).isoformat(),
            controller=loop_cfg.controller,
            action_spec=action,
            baseline_id=compared_baseline_id,
            run_ids=[item.run_id for item in candidate_metrics],
            run_score=run_score,
            resolver_action=resolution.action,
            resolver_reason=resolution.reason,
            depth=loop_cfg.depth,
            width=loop_cfg.width,
            accepted_primary_metric=raw_accepted_metric if resolution.action == "adopt" else None,
        )
        history.append(entry)
        per_iteration_metrics.append(candidate_metrics)
        wisdom.update(action.family, run_score)
        wisdom.save(wisdom_path)

    accepted_metric = adapter.canonical_primary_metric()
    report_dir = report_generator.write_loop_artifacts(
        controller=loop_cfg.controller,
        initial_baseline=initial_baseline,
        final_baseline=current_baseline,
        history=history,
        per_iteration_metrics=per_iteration_metrics,
        wisdom=wisdom,
        final_canonical_metric=accepted_metric,
    )
    (report_dir / "config.json").write_text(json.dumps(to_jsonable(loop_cfg), indent=2, sort_keys=True), encoding="utf-8")
    return LoopResult(
        controller=loop_cfg.controller,
        initial_baseline=initial_baseline,
        final_baseline=current_baseline,
        history=history,
        accepted_metric=accepted_metric,
        raw_accepted_metric=raw_accepted_metric,
        output_dir=report_dir,
    )


def run_comparison(
    output_root: Path,
    cfg: ExperimentConfig | None = None,
    *,
    iterations: int = 4,
    seeds: int = 3,
    mode: str = "default",
) -> dict[str, LoopResult]:
    cfg = cfg or ExperimentConfig()
    output_root.mkdir(parents=True, exist_ok=True)
    shared_progress = RunProgressTracker(total_runs=2 * seeds * (iterations + 1))

    vec3_result = run_single_loop(
        cfg,
        LoopConfig(controller="vec3", max_iterations=iterations, n_seeds=seeds, mode=mode),
        output_root,
        progress=shared_progress,
    )
    binary_result = run_single_loop(
        cfg,
        LoopConfig(controller="binary", max_iterations=iterations, n_seeds=seeds, mode=mode),
        output_root,
        progress=shared_progress,
    )

    report_generator = ReportGenerator(output_root)
    report_generator.write_comparison_report(
        baseline_metric=vec3_result.initial_baseline.metrics.primary_metric,
        vec3_history=vec3_result.history,
        binary_history=binary_result.history,
        vec3_final_metric=vec3_result.accepted_metric,
        binary_final_metric=binary_result.accepted_metric,
        mode=mode,
    )
    return {"vec3": vec3_result, "binary": binary_result}


def run_failure_injection_set(
    output_root: Path,
    cfg: ExperimentConfig | None = None,
    *,
    seeds: int = 1,
) -> list[FailureCaseResult]:
    cfg = cfg or ExperimentConfig()
    output_root.mkdir(parents=True, exist_ok=True)
    adapter = create_benchmark_adapter(cfg)
    report_generator = ReportGenerator(output_root)
    failure_scenarios = get_failure_injection_set(cfg.backend)
    progress = RunProgressTracker(total_runs=seeds * (len(failure_scenarios) + 1))
    baseline = adapter.record_baseline(
        list(range(seeds)),
        progress=progress,
        progress_context=RunProgressContext(controller="failure_injection", phase="baseline"),
    )

    results: list[FailureCaseResult] = []
    for iteration, scenario in enumerate(failure_scenarios, start=1):
        action = ACTION_INDEX[scenario.action_name]
        candidate_metrics, _ = adapter.execute(
            action,
            list(range(seeds)),
            progress=progress,
            progress_context=RunProgressContext(
                controller="failure_injection",
                phase="scenario",
                iteration=iteration,
                action_name=scenario.action_name,
            ),
        )
        run_score, _ = score_run_metrics(candidate_metrics, baseline.metrics, cfg.detector)
        resolution = resolve_vec3(run_score, cfg.detector)
        binary_resolution = resolve_binary(candidate_metrics, baseline.metrics, cfg.detector)
        mean_metric = sum(item.primary_metric for item in candidate_metrics) / len(candidate_metrics)
        matched_signals = all(signal in run_score.fired_signals for signal in scenario.expected_signals)
        matched_resolution = resolution.action == scenario.expected_resolution
        results.append(
            FailureCaseResult(
                scenario_name=scenario.name,
                action_spec=action,
                expected_signals=list(scenario.expected_signals),
                expected_resolution=scenario.expected_resolution,
                narrative=scenario.narrative,
                candidate_metric=round(mean_metric, 5),
                run_score=run_score,
                resolution=resolution,
                binary_resolution=binary_resolution,
                matched_expectation=matched_signals and matched_resolution,
            )
        )

    report_generator.write_failure_injection_report(output_root / "failure_injection", baseline, results)
    return results


def _loop_run_count(loop_cfg: LoopConfig) -> int:
    return loop_cfg.n_seeds * (loop_cfg.max_iterations + 1)
