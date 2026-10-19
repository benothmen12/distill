"""Print a few random examples, to review the data by eye.

Usage:
    python -m distill.data.show data/processed/test.jsonl --n 5
"""

from __future__ import annotations

import argparse
import json
import random

from distill.data.example import read_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Show random examples from a JSONL file.")
    parser.add_argument("path")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    examples = read_examples(args.path)
    rng = random.Random(args.seed)
    for example in rng.sample(examples, min(args.n, len(examples))):
        print(f"--- {example.id} | {example.category} | {example.language} | now: {example.now}")
        print(f"USER: {example.user_message}")
        for call in example.tool_calls:
            print(f"CALL: {call.name}({json.dumps(call.arguments, ensure_ascii=False)})")
        if example.assistant_reply:
            print(f"REPLY: {example.assistant_reply}")
        print()


if __name__ == "__main__":
    main()
