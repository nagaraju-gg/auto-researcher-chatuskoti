from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Vec3:
    truthness: float
    coherence: float
    comparability: float


@dataclass(frozen=True)
class RunMetrics:
    run_id: str
    seed: int
    primary_metric: float
    train_loss: float
    val_loss: float
    train_val_gap: float
    grad_norm_mean: float
    grad_norm_std: float
    weight_distance: float
    param_count: int
    eval_hash: str
    model_family: str
    objective_family: str
    proxy_metrics: dict[str, float]
    detector_inputs: dict[str, float | str | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class SeedScore:
    vec3: Vec3
    goodhart_score: float
    fired_signals: list[str]
    raw_detectors: dict[str, float | str | bool]


@dataclass(frozen=True)
class RunScore:
    mean: Vec3
    std: Vec3
    mag: float
    spread: float
    goodhart_score: float
    fired_signals: list[str]
    raw_detectors: dict[str, float]


@dataclass(frozen=True)
class ActionSpec:
    name: str
    family: str
    params: dict[str, Any]
    rationale: str


@dataclass(frozen=True)
class BaselineRecord:
    baseline_id: str
    metrics: RunMetrics


@dataclass(frozen=True)
class HistoryEntry:
    iteration: int
    timestamp: str
    controller: str
    action_spec: ActionSpec
    baseline_id: str
    run_ids: list[str]
    run_score: RunScore
    resolver_action: str
    resolver_reason: str
    depth: float
    width: int
    accepted_primary_metric: float | None


@dataclass(frozen=True)
class Resolution:
    action: str
    reason: str


@dataclass(frozen=True)
class FailureCaseResult:
    scenario_name: str
    action_spec: ActionSpec
    expected_signals: list[str]
    expected_resolution: str
    narrative: str
    candidate_metric: float
    run_score: RunScore
    resolution: Resolution
    binary_resolution: Resolution
    matched_expectation: bool


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_jsonable(inner) for key, inner in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [to_jsonable(inner) for inner in value]
    if isinstance(value, tuple):
        return [to_jsonable(inner) for inner in value]
    if isinstance(value, Path):
        return str(value)
    return value
