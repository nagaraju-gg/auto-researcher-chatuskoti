from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from catuskoti_ar.config import ExperimentConfig
from catuskoti_ar.runner import run_failure_injection_set


class FailureRunnerTests(unittest.TestCase):
    def test_failure_runner_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "failure_run"
            results = run_failure_injection_set(root, ExperimentConfig(), seeds=1)
            self.assertTrue((root / "failure_injection" / "summary.md").exists())
            self.assertGreaterEqual(len(results), 1)
            self.assertTrue(all(result.binary_resolution.action in {"adopt", "reject"} for result in results))
