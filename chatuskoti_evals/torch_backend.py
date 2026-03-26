from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from statistics import mean

from chatuskoti_evals.config import TorchBenchmarkConfig
from chatuskoti_evals.models import ActionSpec, BaselineRecord, RunMetrics
from chatuskoti_evals.progress import RunProgressContext, RunProgressSnapshot, RunProgressTracker


@dataclass(frozen=True)
class TorchTrainingRecipe:
    optimizer_name: str = "sgd"
    learning_rate: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9
    scheduler_name: str = "multistep"
    warmup_epochs: int = 0
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0
    classifier_dropout: float = 0.0
    stochastic_depth_rate: float = 0.0
    failure_injection: str = ""
    objective_family: str = "cross_entropy"
    eval_tta: bool = False


class TorchCIFAR100ResNet18Adapter:
    """Real training backend for a CIFAR-100/ResNet-18 benchmark.

    The repo still uses the simulator by default because this environment does not
    have torch installed. This adapter is lazy-imported and ready for a real machine.
    """

    def __init__(self, cfg: TorchBenchmarkConfig):
        self.cfg = cfg
        self.current_recipe = TorchTrainingRecipe(
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            momentum=cfg.momentum,
            label_smoothing=cfg.label_smoothing,
        )
        self._metrics_cache: dict[tuple[str, int, str], RunMetrics] = {}
        self._weight_cache: dict[tuple[str, int], list[float]] = {}
        self._dataset_cache: tuple[object, object] | None = None
        self._last_baseline_metric: float = 0.0
        self._import_torch_modules()

    def record_baseline(
        self,
        seeds: list[int],
        *,
        progress: RunProgressTracker | None = None,
        progress_context: RunProgressContext | None = None,
    ) -> BaselineRecord:
        metrics = [
            self._run_recipe_seed(
                self.current_recipe,
                seed,
                baseline_recipe_hash="baseline",
                progress=progress,
                progress_context=progress_context,
            )
            for seed in seeds
        ]
        aggregate = self._aggregate_metrics(metrics, run_id=f"{self._recipe_hash(self.current_recipe)}-mean")
        self._last_baseline_metric = aggregate.primary_metric
        baseline_id = sha256(
            f"{aggregate.eval_hash}:{aggregate.primary_metric:.5f}:{aggregate.train_loss:.5f}".encode("utf-8")
        ).hexdigest()[:12]
        return BaselineRecord(baseline_id=baseline_id, metrics=aggregate)

    def execute(
        self,
        action: ActionSpec,
        seeds: list[int],
        *,
        progress: RunProgressTracker | None = None,
        progress_context: RunProgressContext | None = None,
    ) -> tuple[list[RunMetrics], TorchTrainingRecipe]:
        recipe = self._apply_action(self.current_recipe, action)
        baseline_hash = self._recipe_hash(self.current_recipe)
        metrics = [
            self._run_recipe_seed(
                recipe,
                seed,
                baseline_recipe_hash=baseline_hash,
                progress=progress,
                progress_context=progress_context,
            )
            for seed in seeds
        ]
        return metrics, recipe

    def adopt(self, candidate: TorchTrainingRecipe) -> None:
        self.current_recipe = candidate

    def canonical_primary_metric(self) -> float:
        return self._last_baseline_metric

    def _run_recipe_seed(
        self,
        recipe: TorchTrainingRecipe,
        seed: int,
        baseline_recipe_hash: str,
        *,
        progress: RunProgressTracker | None = None,
        progress_context: RunProgressContext | None = None,
    ) -> RunMetrics:
        recipe_hash = self._recipe_hash(recipe)
        cache_key = (recipe_hash, seed, baseline_recipe_hash)
        progress_snapshot = (
            progress.start_run(progress_context, seed=seed, cache_hit=cache_key in self._metrics_cache)
            if progress
            else None
        )
        if cache_key in self._metrics_cache:
            self._log_run_start(recipe_hash, seed, self._resolve_device(), progress_snapshot)
            if progress is not None:
                progress.finish_run()
            return self._metrics_cache[cache_key]

        torch = self._torch
        nn = self._nn

        self._seed_everything(seed)
        device = self._resolve_device()
        self._log_run_start(recipe_hash, seed, device, progress_snapshot)
        train_loader, val_loader = self._build_dataloaders(seed, device)
        model = self._build_model(recipe, device)
        optimizer = self._build_optimizer(recipe, model)
        scheduler = self._build_scheduler(recipe, optimizer)
        grad_scaler = self._GradScaler(enabled=self.cfg.use_amp and device.type == "cuda")
        total_grad_norms: list[float] = []
        epoch_train_losses: list[float] = []
        epoch_val_losses: list[float] = []

        for epoch in range(self.cfg.epochs):
            model.train()
            total_loss = 0.0
            total_items = 0
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                mixed_images, target_a, target_b, lam = self._maybe_mixup(images, labels, recipe, torch)

                with self._autocast(device):
                    logits = model(mixed_images)
                    if lam is None:
                        loss = self._compute_loss(logits, labels, recipe)
                    else:
                        loss = lam * self._compute_loss(logits, target_a, recipe) + (1.0 - lam) * self._compute_loss(
                            logits, target_b, recipe
                        )

                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf")).item())
                if math.isfinite(grad_norm):
                    total_grad_norms.append(grad_norm)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                total_loss += float(loss.detach().item()) * labels.size(0)
                total_items += int(labels.size(0))

            if scheduler is not None:
                scheduler.step()

            epoch_train_loss = total_loss / max(total_items, 1)
            epoch_train_losses.append(epoch_train_loss)
            eval_result = self._evaluate(model, val_loader, recipe, device)
            epoch_val_losses.append(eval_result["val_loss"])
            self._log(
                f"[torch] run={recipe_hash} seed={seed} epoch={epoch + 1}/{self.cfg.epochs} "
                f"train_loss={epoch_train_loss:.4f} val_loss={eval_result['val_loss']:.4f} "
                f"val_acc={eval_result['accuracy']:.4f}"
            )

        final_eval = self._evaluate(model, val_loader, recipe, device)
        train_loss = epoch_train_losses[-1] if epoch_train_losses else float("nan")
        val_loss = final_eval["val_loss"]
        train_val_gap = val_loss - train_loss
        grad_norm_mean = mean(total_grad_norms) if total_grad_norms else float("nan")
        grad_norm_std = self._std(total_grad_norms)
        param_count = sum(param.numel() for param in model.parameters())

        sample_vector = self._sample_parameter_vector(model)
        self._weight_cache[(recipe_hash, seed)] = sample_vector
        baseline_vector = self._weight_cache.get((baseline_recipe_hash, seed))
        weight_distance = 0.0 if baseline_vector is None else self._cosine_distance(sample_vector, baseline_vector)

        primary_metric = final_eval["accuracy"]
        proxy_metric_corr = final_eval["proxy_metric_corr"]
        calibration = final_eval["calibration"]
        primary_metric, train_loss, val_loss, train_val_gap, grad_norm_mean, grad_norm_std, proxy_metric_corr, calibration = (
            self._inject_failure_metrics(
                recipe=recipe,
                primary_metric=primary_metric,
                train_loss=train_loss,
                val_loss=val_loss,
                train_val_gap=train_val_gap,
                grad_norm_mean=grad_norm_mean,
                grad_norm_std=grad_norm_std,
                proxy_metric_corr=proxy_metric_corr,
                calibration=calibration,
            )
        )

        metric = RunMetrics(
            run_id=f"{recipe_hash}-seed{seed}",
            seed=seed,
            primary_metric=round(primary_metric, 5),
            train_loss=round(train_loss, 5),
            val_loss=round(val_loss, 5),
            train_val_gap=round(train_val_gap, 5),
            grad_norm_mean=round(grad_norm_mean, 5),
            grad_norm_std=round(grad_norm_std, 5),
            weight_distance=round(weight_distance, 5),
            param_count=param_count,
            eval_hash=self._eval_hash(recipe),
            model_family="resnet18",
            objective_family=recipe.objective_family,
            proxy_metrics={
                "proxy_metric_corr": round(proxy_metric_corr, 5),
                "calibration": round(calibration, 5),
            },
            detector_inputs={
                "backend": "torch",
                "regularization_impl": self._regularization_impl(recipe),
                "recipe_hash": recipe_hash,
            },
        )
        self._metrics_cache[cache_key] = metric
        if progress is not None:
            progress.finish_run()
        return metric

    def _apply_action(self, recipe: TorchTrainingRecipe, action: ActionSpec) -> TorchTrainingRecipe:
        if action.name == "pyrrhic_probe":
            return replace(recipe, failure_injection="pyrrhic_probe")
        if action.name == "metric_gaming_probe":
            return replace(recipe, failure_injection="metric_gaming_probe")
        if action.name == "broken_probe":
            return replace(recipe, failure_injection="broken_probe")
        if action.name == "stochastic_depth_high":
            return replace(
                recipe,
                stochastic_depth_rate=max(recipe.stochastic_depth_rate, 0.10),
                label_smoothing=max(recipe.label_smoothing, 0.05),
            )
        if action.name == "stochastic_depth_low":
            return replace(recipe, stochastic_depth_rate=max(recipe.stochastic_depth_rate, 0.08))
        if action.name == "dropout_high":
            return replace(
                recipe,
                classifier_dropout=max(recipe.classifier_dropout, 0.18),
                label_smoothing=max(recipe.label_smoothing, 0.08),
            )
        if action.name == "label_smoothing":
            return replace(recipe, label_smoothing=max(recipe.label_smoothing, float(action.params["label_smoothing"])))
        if action.name == "mixup":
            return replace(recipe, mixup_alpha=max(recipe.mixup_alpha, float(action.params["alpha"])))
        if action.name == "adamw":
            return replace(
                recipe,
                optimizer_name="adamw",
                learning_rate=float(action.params["lr"]),
                weight_decay=float(action.params["weight_decay"]),
            )
        if action.name == "high_lr":
            return replace(recipe, learning_rate=recipe.learning_rate * float(action.params["lr_multiplier"]))
        if action.name == "cosine_warmup":
            return replace(recipe, scheduler_name="cosine", warmup_epochs=max(recipe.warmup_epochs, int(action.params["warmup_epochs"])))
        if action.name == "focal_objective":
            return replace(recipe, objective_family="focal_loss")
        if action.name == "eval_tta":
            return replace(recipe, eval_tta=True)
        return recipe

    def _build_model(self, recipe: TorchTrainingRecipe, device: object) -> object:
        model = self._models.resnet18(weights=None, num_classes=100)
        if recipe.stochastic_depth_rate > 0.0:
            self._apply_residual_stochastic_depth(model, recipe.stochastic_depth_rate)
        if recipe.classifier_dropout > 0.0:
            in_features = model.fc.in_features
            model.fc = self._nn.Sequential(
                self._nn.Dropout(p=recipe.classifier_dropout),
                self._nn.Linear(in_features, 100),
            )
        return model.to(device)

    def _build_optimizer(self, recipe: TorchTrainingRecipe, model: object) -> object:
        if recipe.optimizer_name == "adamw":
            return self._torch.optim.AdamW(model.parameters(), lr=recipe.learning_rate, weight_decay=recipe.weight_decay)
        return self._torch.optim.SGD(
            model.parameters(),
            lr=recipe.learning_rate,
            momentum=recipe.momentum,
            weight_decay=recipe.weight_decay,
        )

    def _build_scheduler(self, recipe: TorchTrainingRecipe, optimizer: object) -> object | None:
        if recipe.scheduler_name == "cosine":
            total_epochs = max(self.cfg.epochs, 1)
            warmup_epochs = max(recipe.warmup_epochs, 0)

            def lr_lambda(epoch: int) -> float:
                if warmup_epochs > 0 and epoch < warmup_epochs:
                    return max((epoch + 1) / warmup_epochs, 1e-3)
                progress_numerator = max(epoch - warmup_epochs, 0)
                progress_denominator = max(total_epochs - warmup_epochs, 1)
                progress = progress_numerator / progress_denominator
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            return self._torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        milestones = [max(self.cfg.epochs // 2, 1), max(int(self.cfg.epochs * 0.75), 1)]
        return self._torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    def _compute_loss(self, logits: object, targets: object, recipe: TorchTrainingRecipe) -> object:
        if recipe.objective_family == "focal_loss":
            log_probs = self._F.log_softmax(logits, dim=1)
            probs = log_probs.exp()
            target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            focal_weight = (1.0 - target_probs) ** 2.0
            return -(focal_weight * target_log_probs).mean()
        return self._F.cross_entropy(logits, targets, label_smoothing=recipe.label_smoothing)

    def _evaluate(self, model: object, val_loader: object, recipe: TorchTrainingRecipe, device: object) -> dict[str, float]:
        torch = self._torch
        model.eval()
        total_loss = 0.0
        total_items = 0
        total_correct = 0
        confidences: list[float] = []
        correctness: list[float] = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                if recipe.eval_tta and self.cfg.tta_horizontal_flip:
                    flipped = torch.flip(images, dims=[3])
                    logits = 0.5 * (logits + model(flipped))

                loss = self._compute_loss(logits, labels, recipe)
                probabilities = self._F.softmax(logits, dim=1)
                confidence, predictions = probabilities.max(dim=1)
                is_correct = predictions.eq(labels)

                batch_size = int(labels.size(0))
                total_loss += float(loss.item()) * batch_size
                total_items += batch_size
                total_correct += int(is_correct.sum().item())
                confidences.extend(float(value) for value in confidence.detach().cpu().tolist())
                correctness.extend(float(value) for value in is_correct.detach().float().cpu().tolist())

        accuracy = total_correct / max(total_items, 1)
        mean_confidence = sum(confidences) / max(len(confidences), 1)
        return {
            "val_loss": total_loss / max(total_items, 1),
            "accuracy": accuracy,
            "proxy_metric_corr": self._pearson(confidences, correctness),
            "calibration": max(0.0, 1.0 - abs(mean_confidence - accuracy)),
        }

    def _build_dataloaders(self, seed: int, device: object) -> tuple[object, object]:
        train_dataset, eval_dataset = self._datasets_for_split()
        train_indices, val_indices = self._split_indices(len(train_dataset))
        generator = self._torch.Generator().manual_seed(seed)
        train_subset = self._Subset(train_dataset, train_indices)
        val_subset = self._Subset(eval_dataset, val_indices)
        num_workers = self._effective_num_workers(device)
        pin_memory = device.type == "cuda"
        train_loader = self._DataLoader(
            train_subset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            generator=generator,
        )
        val_loader = self._DataLoader(
            val_subset,
            batch_size=self.cfg.eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return train_loader, val_loader

    def _datasets_for_split(self) -> tuple[object, object]:
        if self._dataset_cache is not None:
            return self._dataset_cache
        normalize = self._transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
        train_transform = self._transforms.Compose(
            [
                self._transforms.RandomCrop(32, padding=4),
                self._transforms.RandomHorizontalFlip(),
                self._transforms.ToTensor(),
                normalize,
            ]
        )
        eval_transform = self._transforms.Compose(
            [
                self._transforms.ToTensor(),
                normalize,
            ]
        )
        root = Path(self.cfg.data_dir)
        train_dataset = self._datasets.CIFAR100(root=str(root), train=True, download=self.cfg.download, transform=train_transform)
        eval_dataset = self._datasets.CIFAR100(root=str(root), train=True, download=self.cfg.download, transform=eval_transform)
        self._dataset_cache = (train_dataset, eval_dataset)
        return self._dataset_cache

    def _split_indices(self, dataset_size: int) -> tuple[list[int], list[int]]:
        rng = random.Random(self.cfg.split_seed)
        indices = list(range(dataset_size))
        rng.shuffle(indices)
        val_size = max(1, int(dataset_size * self.cfg.val_fraction))
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        return train_indices, val_indices

    def _sample_parameter_vector(self, model: object, max_items: int = 4096) -> list[float]:
        flat = self._torch.cat([param.detach().float().view(-1).cpu() for param in model.parameters()])
        if flat.numel() <= max_items:
            return [float(value) for value in flat.tolist()]
        step = max(flat.numel() // max_items, 1)
        sampled = flat[::step][:max_items]
        return [float(value) for value in sampled.tolist()]

    def _recipe_hash(self, recipe: TorchTrainingRecipe) -> str:
        raw = (
            f"{recipe.optimizer_name}|{recipe.learning_rate}|{recipe.weight_decay}|{recipe.momentum}|"
            f"{recipe.scheduler_name}|{recipe.warmup_epochs}|{recipe.label_smoothing}|{recipe.mixup_alpha}|"
            f"{recipe.classifier_dropout}|{recipe.stochastic_depth_rate}|{recipe.failure_injection}|"
            f"{recipe.objective_family}|{recipe.eval_tta}"
        )
        return sha256(raw.encode("utf-8")).hexdigest()[:12]

    def _eval_hash(self, recipe: TorchTrainingRecipe) -> str:
        raw = (
            f"{self.cfg.dataset}|{self.cfg.model}|split={self.cfg.split_seed}|val_fraction={self.cfg.val_fraction}|"
            f"objective={recipe.objective_family}|tta={recipe.eval_tta}"
        )
        return sha256(raw.encode("utf-8")).hexdigest()[:12]

    def _aggregate_metrics(self, metrics: list[RunMetrics], run_id: str) -> RunMetrics:
        first = metrics[0]
        calc = lambda getter: round(sum(getter(item) for item in metrics) / len(metrics), 5)
        return RunMetrics(
            run_id=run_id,
            seed=-1,
            primary_metric=calc(lambda item: item.primary_metric),
            train_loss=calc(lambda item: item.train_loss),
            val_loss=calc(lambda item: item.val_loss),
            train_val_gap=calc(lambda item: item.train_val_gap),
            grad_norm_mean=calc(lambda item: item.grad_norm_mean),
            grad_norm_std=calc(lambda item: item.grad_norm_std),
            weight_distance=calc(lambda item: item.weight_distance),
            param_count=first.param_count,
            eval_hash=first.eval_hash,
            model_family=first.model_family,
            objective_family=first.objective_family,
            proxy_metrics={
                "proxy_metric_corr": calc(lambda item: item.proxy_metrics["proxy_metric_corr"]),
                "calibration": calc(lambda item: item.proxy_metrics["calibration"]),
            },
            detector_inputs={"backend": "torch"},
        )

    def _seed_everything(self, seed: int) -> None:
        random.seed(seed)
        self._torch.manual_seed(seed)
        if self._torch.cuda.is_available():
            self._torch.cuda.manual_seed_all(seed)

    def _resolve_device(self) -> object:
        torch = self._torch
        if self.cfg.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.cfg.device)

    def _effective_num_workers(self, device: object) -> int:
        if getattr(device, "type", "cpu") == "mps":
            return 0
        return max(self.cfg.num_workers, 0)

    def _maybe_mixup(self, images: object, labels: object, recipe: TorchTrainingRecipe, torch: object) -> tuple[object, object, object, float | None]:
        if recipe.mixup_alpha <= 0.0:
            return images, labels, labels, None
        lam = random.betavariate(recipe.mixup_alpha, recipe.mixup_alpha)
        permutation = torch.randperm(images.size(0), device=images.device)
        mixed = lam * images + (1.0 - lam) * images[permutation]
        return mixed, labels, labels[permutation], lam

    def _autocast(self, device: object):
        if self.cfg.use_amp and getattr(device, "type", "cpu") == "cuda":
            return self._autocast_impl(device_type="cuda", dtype=self._torch.float16)
        return self._nullcontext()

    def _import_torch_modules(self) -> None:
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from contextlib import nullcontext
            from torch.amp import GradScaler, autocast
            from torch.utils.data import DataLoader, Subset
            from torchvision import datasets, models, transforms
        except ModuleNotFoundError as exc:
            raise ImportError(
                "Torch backend requires torch and torchvision. Install `requirements-torch.txt` on a GPU-capable machine."
            ) from exc

        self._torch = torch
        self._nn = nn
        self._F = F
        self._nullcontext = nullcontext
        self._GradScaler = GradScaler
        self._autocast_impl = autocast
        self._DataLoader = DataLoader
        self._Subset = Subset
        self._datasets = datasets
        self._models = models
        self._transforms = transforms

    def _log(self, message: str) -> None:
        if self.cfg.log_every_epoch:
            print(message, flush=True)

    def _log_run_start(
        self,
        recipe_hash: str,
        seed: int,
        device: object,
        progress_snapshot: RunProgressSnapshot | None,
    ) -> None:
        parts = ["[torch]"]
        if progress_snapshot is not None:
            parts.append(
                "progress="
                f"{progress_snapshot.current_run}/{progress_snapshot.total_runs}"
            )
            parts.append(f"remaining={progress_snapshot.remaining_runs}")
            parts.append(f"stage={progress_snapshot.context.label}")
            if progress_snapshot.cache_hit:
                parts.append("cache=hit")
        parts.append(f"run={recipe_hash}")
        parts.append(f"seed={seed}")
        parts.append(f"device={device.type}")
        parts.append(f"workers={self._effective_num_workers(device)}")
        parts.append(f"epochs={self.cfg.epochs}")
        self._log(" ".join(parts))

    def _regularization_impl(self, recipe: TorchTrainingRecipe) -> str:
        parts: list[str] = []
        if recipe.failure_injection:
            parts.append(recipe.failure_injection)
        if recipe.stochastic_depth_rate > 0.0:
            parts.append("residual_stochastic_depth")
        if recipe.classifier_dropout > 0.0:
            parts.append("classifier_dropout")
        if not parts:
            return "none"
        return "+".join(parts)

    def _inject_failure_metrics(
        self,
        *,
        recipe: TorchTrainingRecipe,
        primary_metric: float,
        train_loss: float,
        val_loss: float,
        train_val_gap: float,
        grad_norm_mean: float,
        grad_norm_std: float,
        proxy_metric_corr: float,
        calibration: float,
    ) -> tuple[float, float, float, float, float, float, float, float]:
        if recipe.failure_injection == "pyrrhic_probe":
            primary_metric += 0.03
            val_loss = max(val_loss, train_loss + 0.32)
            train_val_gap = val_loss - train_loss
            grad_norm_std = max(grad_norm_std, 1.9)
            calibration = max(0.0, calibration - 0.12)
        elif recipe.failure_injection == "metric_gaming_probe":
            primary_metric += 0.035
            grad_norm_std = min(grad_norm_std, 0.01)
            proxy_metric_corr = max(0.02, proxy_metric_corr - 0.40)
            calibration = max(0.0, calibration - 0.08)
        elif recipe.failure_injection == "broken_probe":
            primary_metric -= 0.03
            val_loss = max(val_loss, train_loss + 0.36)
            train_val_gap = val_loss - train_loss
            grad_norm_mean = max(grad_norm_mean, 8.5)
            grad_norm_std = max(grad_norm_std, 2.1)
            calibration = max(0.0, calibration - 0.18)
        return (
            primary_metric,
            train_loss,
            val_loss,
            train_val_gap,
            grad_norm_mean,
            grad_norm_std,
            proxy_metric_corr,
            calibration,
        )

    def _apply_residual_stochastic_depth(self, model: object, max_drop_rate: float) -> None:
        blocks: list[tuple[object, int, object]] = []
        for layer_name in ("layer1", "layer2", "layer3", "layer4"):
            layer = getattr(model, layer_name)
            for index, block in enumerate(layer):
                blocks.append((layer, index, block))

        total_blocks = max(len(blocks), 1)
        for block_index, (layer, index, block) in enumerate(blocks, start=1):
            drop_prob = max_drop_rate * block_index / total_blocks
            layer[index] = self._make_drop_path_block(block, drop_prob)

    def _make_drop_path_block(self, block: object, drop_prob: float) -> object:
        nn = self._nn
        torch = self._torch

        class ResidualDropPathBlock(nn.Module):
            def __init__(self, inner_block: object, rate: float):
                super().__init__()
                self.inner_block = inner_block
                self.drop_prob = max(0.0, min(0.95, float(rate)))

            def forward(self, x: object) -> object:
                identity = x

                out = self.inner_block.conv1(x)
                out = self.inner_block.bn1(out)
                out = self.inner_block.relu(out)

                out = self.inner_block.conv2(out)
                out = self.inner_block.bn2(out)

                if self.inner_block.downsample is not None:
                    identity = self.inner_block.downsample(x)

                if self.training and self.drop_prob > 0.0:
                    keep_prob = 1.0 - self.drop_prob
                    shape = (out.shape[0],) + (1,) * (out.ndim - 1)
                    random_tensor = keep_prob + torch.rand(shape, dtype=out.dtype, device=out.device)
                    binary_tensor = random_tensor.floor()
                    out = out / keep_prob * binary_tensor

                out = out + identity
                out = self.inner_block.relu(out)
                return out

        return ResidualDropPathBlock(block, drop_prob)

    @staticmethod
    def _pearson(xs: list[float], ys: list[float]) -> float:
        if len(xs) != len(ys) or len(xs) < 2:
            return 0.0
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        var_x = sum((x - mean_x) ** 2 for x in xs)
        var_y = sum((y - mean_y) ** 2 for y in ys)
        denom = math.sqrt(var_x * var_y)
        if denom <= 1e-12:
            return 0.0
        return max(-1.0, min(1.0, cov / denom))

    @staticmethod
    def _std(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mu = sum(values) / len(values)
        return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))

    @staticmethod
    def _cosine_distance(xs: list[float], ys: list[float]) -> float:
        if not xs or not ys:
            return 0.0
        length = min(len(xs), len(ys))
        dot = sum(xs[index] * ys[index] for index in range(length))
        norm_x = math.sqrt(sum(xs[index] ** 2 for index in range(length)))
        norm_y = math.sqrt(sum(ys[index] ** 2 for index in range(length)))
        if norm_x <= 1e-12 or norm_y <= 1e-12:
            return 0.0
        cosine_similarity = dot / (norm_x * norm_y)
        return max(0.0, min(2.0, 1.0 - cosine_similarity))
