from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DetectorConfig:
    truth_delta_scale: float = 0.02
    min_magnitude: float = 0.18
    max_spread: float = 0.30
    adopt_truth_threshold: float = 0.25
    reject_truth_threshold: float = -0.25
    incoherence_threshold: float = 0.35
    comparability_threshold: float = 0.80
    goodhart_threshold: float = 0.65
    gap_multiplier: float = 1.6
    grad_mean_multiplier: float = 1.8
    grad_std_multiplier: float = 1.7
    hypercoherence_ratio: float = 0.40
    proxy_corr_drop: float = 0.18
    improvement_for_goodhart: float = 0.015
    binary_metric_threshold: float = 0.002


@dataclass(frozen=True)
class LoopConfig:
    controller: str
    max_iterations: int = 4
    n_seeds: int = 3
    width: int = 1
    depth: float = 1.0
    mode: str = "default"


@dataclass(frozen=True)
class SimulationConfig:
    dataset: str = "CIFAR-100"
    model: str = "ResNet-18"
    framework: str = "PyTorch-compatible simulation"
    baseline_train_loss: float = 1.45
    baseline_primary_metric: float = 0.636
    param_count: int = 11_200_000


@dataclass(frozen=True)
class TorchBenchmarkConfig:
    dataset: str = "CIFAR-100"
    model: str = "ResNet-18"
    framework: str = "PyTorch"
    data_dir: Path = Path("data")
    device: str = "auto"
    epochs: int = 3
    batch_size: int = 128
    eval_batch_size: int = 256
    num_workers: int = 2
    learning_rate: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9
    val_fraction: float = 0.1
    split_seed: int = 13
    download: bool = True
    use_amp: bool = True
    label_smoothing: float = 0.0
    tta_horizontal_flip: bool = True
    log_every_epoch: bool = True


@dataclass(frozen=True)
class ReportConfig:
    output_dir: Path
    keep_raw_seed_metrics: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    backend: str = "simulator"
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    torch: TorchBenchmarkConfig = field(default_factory=TorchBenchmarkConfig)
