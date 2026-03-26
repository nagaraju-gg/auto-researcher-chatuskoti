from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from catuskoti_ar.models import RunScore, Vec3


class WisdomStore:
    def __init__(self) -> None:
        self._counts: dict[str, int] = defaultdict(int)
        self._truthness: dict[str, float] = defaultdict(float)
        self._coherence: dict[str, float] = defaultdict(float)
        self._comparability: dict[str, float] = defaultdict(float)
        self._goodhart: dict[str, float] = defaultdict(float)

    def update(self, family: str, run_score: RunScore) -> None:
        n = self._counts[family] + 1
        self._truthness[family] = running_mean(self._truthness[family], run_score.mean.truthness, n)
        self._coherence[family] = running_mean(self._coherence[family], run_score.mean.coherence, n)
        self._comparability[family] = running_mean(self._comparability[family], run_score.mean.comparability, n)
        self._goodhart[family] = running_mean(self._goodhart[family], run_score.goodhart_score, n)
        self._counts[family] = n

    def predict(self, family: str) -> Vec3:
        return Vec3(
            truthness=self._truthness[family],
            coherence=self._coherence[family],
            comparability=self._comparability[family],
        )

    def family_score(self, family: str) -> float:
        vec = self.predict(family)
        risk = self._goodhart[family]
        return vec.truthness + 0.35 * vec.coherence + 0.10 * vec.comparability - 0.60 * risk

    def confident_families(self, min_seen: int = 2) -> list[str]:
        return sorted(family for family, count in self._counts.items() if count >= min_seen)

    def save(self, path: Path) -> None:
        payload = {
            "counts": dict(self._counts),
            "truthness": dict(self._truthness),
            "coherence": dict(self._coherence),
            "comparability": dict(self._comparability),
            "goodhart": dict(self._goodhart),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "WisdomStore":
        store = cls()
        if not path.exists():
            return store
        payload = json.loads(path.read_text(encoding="utf-8"))
        for family, count in payload.get("counts", {}).items():
            store._counts[family] = int(count)
        for family, value in payload.get("truthness", {}).items():
            store._truthness[family] = float(value)
        for family, value in payload.get("coherence", {}).items():
            store._coherence[family] = float(value)
        for family, value in payload.get("comparability", {}).items():
            store._comparability[family] = float(value)
        for family, value in payload.get("goodhart", {}).items():
            store._goodhart[family] = float(value)
        return store

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        return {
            family: {
                "count": self._counts[family],
                "truthness": round(self._truthness[family], 5),
                "coherence": round(self._coherence[family], 5),
                "comparability": round(self._comparability[family], 5),
                "goodhart": round(self._goodhart[family], 5),
                "score": round(self.family_score(family), 5),
            }
            for family in sorted(self._counts)
        }


def running_mean(previous: float, value: float, n: int) -> float:
    return previous + (value - previous) / n

