from __future__ import annotations

import json
import sys
from pathlib import Path

from chatuskoti_evals.reporting import write_challenge_table_markdown, write_challenge_table_svg


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python3 scripts/generate_challenge_table.py <challenge_compare_dir>")
        return 1

    root = Path(sys.argv[1])
    vec3_history = load_history(root / "vec3" / "history.jsonl")
    binary_history = load_history(root / "binary" / "history.jsonl")
    write_challenge_table_markdown(root / "challenge_cases.md", vec3_history, binary_history)
    write_challenge_table_svg(root / "challenge_cases.svg", vec3_history, binary_history)
    print(f"Wrote challenge case artifacts to {root}")
    return 0


def load_history(path: Path) -> list:
    from chatuskoti_evals.models import ActionSpec, HistoryEntry, RunScore, Vec3

    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        run_score = payload["run_score"]
        entry = HistoryEntry(
            iteration=int(payload["iteration"]),
            timestamp=payload["timestamp"],
            controller=payload["controller"],
            action_spec=ActionSpec(
                name=payload["action_spec"]["name"],
                family=payload["action_spec"]["family"],
                params=payload["action_spec"]["params"],
                rationale=payload["action_spec"]["rationale"],
            ),
            baseline_id=payload["baseline_id"],
            run_ids=list(payload["run_ids"]),
            run_score=RunScore(
                mean=Vec3(**run_score["mean"]),
                std=Vec3(**run_score["std"]),
                mag=run_score["mag"],
                spread=run_score["spread"],
                goodhart_score=run_score["goodhart_score"],
                fired_signals=list(run_score["fired_signals"]),
                raw_detectors=run_score["raw_detectors"],
            ),
            resolver_action=payload["resolver_action"],
            resolver_reason=payload["resolver_reason"],
            depth=payload["depth"],
            width=payload["width"],
            accepted_primary_metric=payload["accepted_primary_metric"],
        )
        entries.append(entry)
    return entries


if __name__ == "__main__":
    raise SystemExit(main())
