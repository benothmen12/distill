"""Compare models side by side, on clean and noisy test sets.

Usage:
    python -m distill.eval.compare results/teacher results/qwen3-4b-base results/qwen3-4b-sft

Prints a Markdown table, ready for the README, and writes results/comparison.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from distill.eval.report import headline_table


def collect(model_dirs: list[Path]) -> dict[str, dict]:
    columns: dict[str, dict] = {}
    for model_dir in model_dirs:
        for split in ("test", "test_noisy"):
            path = model_dir / split / "metrics.json"
            if path.exists():
                label = model_dir.name if split == "test" else f"{model_dir.name} (noisy)"
                columns[label] = json.loads(path.read_text(encoding="utf-8"))
    return columns


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare evaluation results.")
    parser.add_argument("models", nargs="+", type=Path, help="results/<name> folders")
    parser.add_argument("--out", type=Path, default=Path("results/comparison.md"))
    args = parser.parse_args()

    columns = collect(args.models)
    if not columns:
        raise SystemExit("No metrics.json found. Run distill.eval.run first.")
    table = headline_table(columns)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
