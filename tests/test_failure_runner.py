from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from chatuskoti_evals.config import ExperimentConfig
from chatuskoti_evals.runner import run_ablation_bundle, run_failure_injection_set


class FailureRunnerTests(unittest.TestCase):
    def test_failure_runner_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "failure_run"
            results = run_failure_injection_set(root, ExperimentConfig(), seeds=1)
            self.assertTrue((root / "failure_injection" / "summary.md").exists())
            self.assertTrue((root / "manifest.json").exists())
            self.assertGreaterEqual(len(results), 1)
            self.assertTrue(all(result.binary_resolution.action in {"adopt", "reject"} for result in results))

    def test_ablation_bundle_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "ablation_run"
            summaries = run_ablation_bundle(root, ExperimentConfig(), seeds=1)
            self.assertEqual(len(summaries), 6)
            self.assertTrue((root / "summary.md").exists())
            self.assertTrue((root / "summary.json").exists())
            self.assertTrue((root / "manifest.json").exists())
