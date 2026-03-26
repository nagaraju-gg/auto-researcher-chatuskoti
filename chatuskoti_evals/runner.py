from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, pstdev

from chatuskoti_evals.benchmark import create_benchmark_adapter
from chatuskoti_evals.config import AblationConfig, ExperimentConfig, LoopConfig
from chatuskoti_evals.actions import ACTION_INDEX
from chatuskoti_evals.models import AggregateSummary, BaselineRecord, BundleManifest, FailureCaseResult, HistoryEntry, RunMetrics, to_jsonable
from chatuskoti_evals.progress import RunProgressContext, RunProgressTracker
from chatuskoti_evals.proposals import ProposalEngine
from chatuskoti_evals.reporting import ReportGenerator, aggregate_failure_results
from chatuskoti_evals.resolver import resolve_binary, resolve_vec3
from chatuskoti_evals.scenarios import get_failure_injection_set
from chatuskoti_evals.scoring import score_run_metrics
from chatuskoti_evals.wisdom import WisdomStore


@dataclass(frozen=True)
class LoopResult:
    controller: str
    initial_baseline: BaselineRecord
    final_baseline: BaselineRecord
    history: list[HistoryEntry]
    accepted_metric: float
    raw_accepted_metric: float
    output_dir: Path


@dataclass(frozen=True)
class FailureCaseExecution:
    scenario_name: str
    action_name: str
    expected_signals: list[str]
    expected_resolution: str
    narrative: str
    candidate_metrics: list[RunMetrics]


def run_single_loop(
    cfg: ExperimentConfig,
    loop_cfg: LoopConfig,
    output_root: Path,
    *,
    progress: RunProgressTracker | None = None,
) -> LoopResult:
    detector_cfg = cfg.ablation.apply(cfg.detector)
    adapter = create_benchmark_adapter(cfg)
    proposal_engine = ProposalEngine()
    report_generator = ReportGenerator(output_root)
    controller_root = output_root / loop_cfg.controller
    controller_root.mkdir(parents=True, exist_ok=True)

    wisdom_path = controller_root / "wisdom_store.json"
    wisdom = WisdomStore.load(wisdom_path) if cfg.ablation.wisdom_enabled else WisdomStore()
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
        active_wisdom = wisdom if cfg.ablation.wisdom_enabled else WisdomStore()
        run_score, _ = score_run_metrics(candidate_metrics, current_baseline.metrics, detector_cfg)

        if loop_cfg.controller == "vec3":
            resolution = resolve_vec3(run_score, detector_cfg)
        else:
            resolution = resolve_binary(candidate_metrics, current_baseline.metrics, detector_cfg)

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
        if cfg.ablation.wisdom_enabled:
            active_wisdom.update(action.family, run_score)
            active_wisdom.save(wisdom_path)
            wisdom = active_wisdom

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
    (report_dir / "config.json").write_text(
        json.dumps(
            {
                "loop": to_jsonable(loop_cfg),
                "detector": to_jsonable(detector_cfg),
                "ablation": cfg.ablation.name,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
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
    _write_manifest(
        output_root / "manifest.json",
        BundleManifest(
            bundle_name=output_root.name,
            artifact_kind="comparison",
            benchmark=_benchmark_name(cfg),
            backend=cfg.backend,
            seeds=seeds,
            epochs=_epoch_count(cfg),
            controller_mode=mode,
            ablation=cfg.ablation.name,
            artifact_paths={
                "comparison": "comparison.md",
                "summary_json": "comparison_summary.json",
                "controller_svg": "controller_comparison.svg",
                "challenge_table": "challenge_cases.md" if mode == "challenge" else "",
            },
        ),
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
    detector_cfg = cfg.ablation.apply(cfg.detector)
    report_generator = ReportGenerator(output_root)
    baseline, executions = _collect_failure_case_executions(cfg, seeds)
    results = _score_failure_case_executions(executions, baseline, detector_cfg)
    report_dir = output_root / "failure_injection"
    report_generator.write_failure_injection_report(report_dir, baseline, results)
    _write_manifest(
        output_root / "manifest.json",
        BundleManifest(
            bundle_name=output_root.name,
            artifact_kind="failure_injection",
            benchmark=_benchmark_name(cfg),
            backend=cfg.backend,
            seeds=seeds,
            epochs=_epoch_count(cfg),
            controller_mode="failure_injection",
            ablation=cfg.ablation.name,
            artifact_paths={
                "summary": "failure_injection/summary.md",
                "summary_json": "failure_injection/aggregate_summary.json",
                "results_json": "failure_injection/failure_results.json",
            },
        ),
    )
    return results


def run_ablation_bundle(
    output_root: Path,
    cfg: ExperimentConfig | None = None,
    *,
    seeds: int = 3,
    ablations: tuple[str, ...] = ("full", "no_coherence", "no_comparability", "no_goodhart", "no_wisdom", "no_spread_gate"),
) -> list[AggregateSummary]:
    cfg = cfg or ExperimentConfig()
    output_root.mkdir(parents=True, exist_ok=True)
    baseline, executions = _collect_failure_case_executions(cfg, seeds)
    report_generator = ReportGenerator(output_root)
    summaries: list[AggregateSummary] = []

    for ablation_name in ablations:
        ablated_cfg = replace(cfg, ablation=AblationConfig(name=ablation_name))
        detector_cfg = ablated_cfg.ablation.apply(ablated_cfg.detector)
        results = _score_failure_case_executions(executions, baseline, detector_cfg)
        report_generator.write_failure_injection_report(output_root / ablation_name / "failure_injection", baseline, results)
        summaries.append(aggregate_failure_results(ablation_name, results))

    report_generator.write_ablation_report(output_root, summaries)
    _write_manifest(
        output_root / "manifest.json",
        BundleManifest(
            bundle_name=output_root.name,
            artifact_kind="ablation_bundle",
            benchmark=_benchmark_name(cfg),
            backend=cfg.backend,
            seeds=seeds,
            epochs=_epoch_count(cfg),
            controller_mode="failure_injection",
            ablation="bundle",
            artifact_paths={
                "summary": "summary.md",
                "summary_json": "summary.json",
                "summary_svg": "ablation_summary.svg",
            },
        ),
    )
    return summaries


def _collect_failure_case_executions(cfg: ExperimentConfig, seeds: int) -> tuple[BaselineRecord, list[FailureCaseExecution]]:
    adapter = create_benchmark_adapter(cfg)
    failure_scenarios = get_failure_injection_set(cfg.backend)
    progress = RunProgressTracker(total_runs=seeds * (len(failure_scenarios) + 1))
    baseline = adapter.record_baseline(
        list(range(seeds)),
        progress=progress,
        progress_context=RunProgressContext(controller="failure_injection", phase="baseline"),
    )
    executions: list[FailureCaseExecution] = []
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
        executions.append(
            FailureCaseExecution(
                scenario_name=scenario.name,
                action_name=scenario.action_name,
                expected_signals=list(scenario.expected_signals),
                expected_resolution=scenario.expected_resolution,
                narrative=scenario.narrative,
                candidate_metrics=candidate_metrics,
            )
        )
    return baseline, executions


def _score_failure_case_executions(
    executions: list[FailureCaseExecution],
    baseline: BaselineRecord,
    detector_cfg,
) -> list[FailureCaseResult]:
    results: list[FailureCaseResult] = []
    for execution in executions:
        action = ACTION_INDEX[execution.action_name]
        run_score, _ = score_run_metrics(execution.candidate_metrics, baseline.metrics, detector_cfg)
        resolution = resolve_vec3(run_score, detector_cfg)
        binary_resolution = resolve_binary(execution.candidate_metrics, baseline.metrics, detector_cfg)
        mean_metric = sum(item.primary_metric for item in execution.candidate_metrics) / len(execution.candidate_metrics)
        matched_signals = all(signal in run_score.fired_signals for signal in execution.expected_signals)
        matched_resolution = resolution.action == execution.expected_resolution
        results.append(
            FailureCaseResult(
                scenario_name=execution.scenario_name,
                action_spec=action,
                expected_signals=list(execution.expected_signals),
                expected_resolution=execution.expected_resolution,
                narrative=execution.narrative,
                candidate_metric=round(mean_metric, 5),
                run_score=run_score,
                resolution=resolution,
                binary_resolution=binary_resolution,
                matched_expectation=matched_signals and matched_resolution,
            )
        )
    return results


def summarize_loop_results(label: str, results: dict[str, LoopResult]) -> dict[str, float | str]:
    vec3 = results["vec3"]
    binary = results["binary"]
    return {
        "label": label,
        "vec3_final_metric": round(vec3.accepted_metric, 5),
        "binary_final_metric": round(binary.accepted_metric, 5),
        "vec3_history_length": len(vec3.history),
        "binary_history_length": len(binary.history),
    }


def _write_manifest(path: Path, manifest: BundleManifest) -> None:
    path.write_text(json.dumps(to_jsonable(manifest), indent=2, sort_keys=True), encoding="utf-8")


def _benchmark_name(cfg: ExperimentConfig) -> str:
    if cfg.backend == "torch":
        return f"{cfg.torch.dataset} + {cfg.torch.model}"
    return f"{cfg.simulation.dataset} + {cfg.simulation.model}"


def _epoch_count(cfg: ExperimentConfig) -> int:
    return cfg.torch.epochs if cfg.backend == "torch" else 0


def _loop_run_count(loop_cfg: LoopConfig) -> int:
    return loop_cfg.n_seeds * (loop_cfg.max_iterations + 1)
