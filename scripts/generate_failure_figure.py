from __future__ import annotations

import json
import sys
from pathlib import Path


ACTION_COLORS = {
    "adopt": "#2563eb",
    "hold": "#f59e0b",
    "reject": "#dc2626",
    "rollback": "#7c3aed",
    "reframe": "#059669",
}


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python3 scripts/generate_failure_figure.py <failure_results.json>")
        return 1

    source = Path(sys.argv[1])
    results = json.loads(source.read_text(encoding="utf-8"))
    out_dir = source.parent
    write_svg(out_dir / "benchmark_figure.svg", results)
    write_caption(out_dir / "benchmark_figure_caption.md", results)
    print(f"Wrote figure assets to {out_dir}")
    return 0


def write_svg(path: Path, results: list[dict]) -> None:
    width = 1160
    height = 460
    left = 34
    top = 28
    row_h = 86
    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="26" font-family="Helvetica" font-size="22" font-weight="700">Vec3 vs Binary on the Canonical Failure Benchmark</text>',
        f'<text x="{left}" y="50" font-family="Helvetica" font-size="12" fill="#444">Torch-backed adversarial calibration suite, CIFAR-100 + ResNet-18, 10 epochs, 1 seed</text>',
        '<line x1="34" y1="68" x2="1124" y2="68" stroke="#d1d5db"/>',
        '<text x="42" y="92" font-family="Helvetica" font-size="12" font-weight="700">Case</text>',
        '<text x="282" y="92" font-family="Helvetica" font-size="12" font-weight="700">Metric</text>',
        '<text x="392" y="92" font-family="Helvetica" font-size="12" font-weight="700">Binary</text>',
        '<text x="532" y="92" font-family="Helvetica" font-size="12" font-weight="700">Vec3</text>',
        '<text x="672" y="92" font-family="Helvetica" font-size="12" font-weight="700">Key Signals</text>',
        '<text x="950" y="92" font-family="Helvetica" font-size="12" font-weight="700">Interpretation</text>',
    ]

    for index, item in enumerate(results):
        y = 104 + index * row_h
        case = simplify_name(item["scenario_name"])
        metric = item["candidate_metric"]
        binary_action = item["binary_resolution"]["action"]
        vec3_action = item["resolution"]["action"]
        signals = ", ".join(item["run_score"]["fired_signals"]) or "none"
        interpretation = interpretation_text(binary_action, vec3_action)

        lines.extend(
            [
                f'<rect x="34" y="{y-18}" width="1090" height="64" fill={"#f8fafc" if index % 2 == 0 else "#ffffff"} stroke="#e5e7eb"/>',
                f'<text x="42" y="{y}" font-family="Helvetica" font-size="13" font-weight="700">{escape(case)}</text>',
                f'<text x="42" y="{y+18}" font-family="Helvetica" font-size="11" fill="#555">{escape(item["action_spec"]["name"])}</text>',
                f'<text x="282" y="{y}" font-family="Helvetica" font-size="13">{metric:.4f}</text>',
                action_badge(392, y - 14, binary_action),
                action_badge(532, y - 14, vec3_action),
                f'<text x="672" y="{y}" font-family="Helvetica" font-size="12">{escape(signals)}</text>',
                f'<text x="950" y="{y}" font-family="Helvetica" font-size="12">{escape(interpretation)}</text>',
            ]
        )

    lines.extend(
        [
            '<line x1="34" y1="426" x2="1124" y2="426" stroke="#d1d5db"/>',
            '<text x="34" y="446" font-family="Helvetica" font-size="12" fill="#444">Headline: binary would adopt 3/4 cases; Vec3 would adopt 0/4 and routes them to hold, reject, rollback, or reframe.</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def action_badge(x: int, y: int, action: str) -> str:
    color = ACTION_COLORS[action]
    label = action.upper()
    width = max(72, 10 * len(label))
    return (
        f'<rect x="{x}" y="{y}" rx="8" ry="8" width="{width}" height="24" fill="{color}" />'
        f'<text x="{x + 10}" y="{y + 16}" font-family="Helvetica" font-size="12" font-weight="700" fill="#ffffff">{label}</text>'
    )


def simplify_name(name: str) -> str:
    mapping = {
        "pyrrhic_probe_injection": "Pyrrhic Probe",
        "goodhart_probe_injection": "Goodhart Probe",
        "broken_failure_injection": "Broken Probe",
        "incomparable_eval_injection": "Eval Regime Shift",
    }
    return mapping.get(name, name.replace("_", " ").title())


def interpretation_text(binary_action: str, vec3_action: str) -> str:
    if binary_action == "adopt" and vec3_action != "adopt":
        return f"Binary false positive; Vec3 {vec3_action}"
    if binary_action == vec3_action:
        return f"Both {vec3_action}"
    return f"Binary {binary_action}; Vec3 {vec3_action}"


def write_caption(path: Path, results: list[dict]) -> None:
    adopts = sum(1 for item in results if item["binary_resolution"]["action"] == "adopt")
    body = [
        "Figure caption:",
        "",
        f"`Vec3` versus binary evaluation on the canonical four-case failure benchmark. "
        f"A metric-only binary controller would adopt `{adopts}` of `{len(results)}` cases, "
        "including the pyrrhic, Goodhart-style, and incomparable cases. "
        "`Vec3` instead routes those cases to `hold`, `reject`, and `reframe`, and escalates the damaged failure case to `rollback`.",
    ]
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


if __name__ == "__main__":
    raise SystemExit(main())
