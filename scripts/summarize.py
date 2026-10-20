"""Collect results/runs/*/metrics_test.json into results/summary.md (the table the README links to).

python scripts/summarize.py
"""

from __future__ import annotations

import json
from pathlib import Path

COLS = [
    ("mae", "MAE (d)"),
    ("crps", "CRPS"),
    ("coverage80", "80% cov."),
    ("width80", "80% width"),
    ("cindex", "C-index"),
    ("ece", "ECE"),
]


def row(name: str, m: dict) -> str:
    return (
        f"| {name} | " + " | ".join(f"{m[k]:.3f}" if m.get(k) == m.get(k) else "—" for k, _ in COLS) + f" | {m['n']} |"
    )


def main() -> None:
    runs = sorted(Path("results/runs").glob("*/metrics_test.json"))
    lines = ["# Results (test split, all unspoiled prefixes of every test item)", ""]
    if not runs:
        lines.append("_No runs yet._")
    for p in runs:
        m = json.loads(p.read_text())
        lines += [
            f"## {p.parent.name}",
            "",
            "| split | " + " | ".join(c for _, c in COLS) + " | n |",
            "|---|" + "---|" * (len(COLS) + 1),
        ]
        for k, v in m.items():
            lines.append(row(k, v))
        lines.append("")
    Path("results/summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
