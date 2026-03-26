from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from catuskoti_ar.config import ExperimentConfig
from catuskoti_ar.runner import run_comparison


class RunnerTests(unittest.TestCase):
    def test_comparison_produces_reports_and_vec3_beats_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "artifacts"
            results = run_comparison(root, ExperimentConfig())

            self.assertTrue((root / "vec3" / "summary.md").exists())
            self.assertTrue((root / "binary" / "summary.md").exists())
            self.assertTrue((root / "comparison.md").exists())

            self.assertGreater(results["vec3"].accepted_metric, results["binary"].accepted_metric)
            vec3_signals = {signal for entry in results["vec3"].history for signal in entry.run_score.fired_signals}
            self.assertIn("hyper_coherence", vec3_signals)

