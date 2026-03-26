from __future__ import annotations

import unittest
from pathlib import Path


class PublicSurfaceTests(unittest.TestCase):
    def test_public_files_use_eval_namespace(self) -> None:
        root = Path(__file__).resolve().parents[1]
        legacy_namespace = "catuskoti" + "_ar"
        legacy_brand = "Catuskoti " + "AutoResearcher"
        checked = list((root / "docs").glob("*.md"))
        checked.extend((root / "scripts").glob("*.sh"))
        checked.extend((root / "scripts").glob("*.py"))
        checked.extend((root / "tests").glob("*.py"))
        checked.extend(
            [
                root / "README.md",
                root / "pyproject.toml",
                root / "CITATION.cff",
            ]
        )
        for path in checked:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn(legacy_namespace, text, path.as_posix())
            self.assertNotIn(legacy_brand, text, path.as_posix())
