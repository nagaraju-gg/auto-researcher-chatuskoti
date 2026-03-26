from __future__ import annotations

import importlib.util
import unittest

from catuskoti_ar.benchmark import SimulatedCIFAR100ResNet18Adapter, create_benchmark_adapter
from catuskoti_ar.config import ExperimentConfig


class BackendFactoryTests(unittest.TestCase):
    def test_simulator_backend_factory(self) -> None:
        adapter = create_benchmark_adapter(ExperimentConfig())
        self.assertIsInstance(adapter, SimulatedCIFAR100ResNet18Adapter)

    def test_torch_backend_error_is_helpful_when_dependencies_are_missing(self) -> None:
        if importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torchvision") is not None:
            self.skipTest("torch dependencies are available in this environment")
        with self.assertRaises(ImportError) as ctx:
            create_benchmark_adapter(ExperimentConfig(backend="torch"))
        self.assertIn("requirements-torch.txt", str(ctx.exception))

