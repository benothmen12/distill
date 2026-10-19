"""Build the final train and test splits from the generated examples.

Usage:
    python -m distill.data.build                 # lexical checks only
    python -m distill.data.build --embeddings    # + semantic checks with embeddings

Steps:
1. Remove exact and near duplicates. Lexically (word 3-grams), and optionally by
   meaning with multilingual embeddings, among examples that expect the same answer.
2. Split train/test, stratified by category and tools, so every tool is tested.
3. Leakage check: a test example too close to a training example is moved to
   train. Otherwise the test score would partly measure memorization.
4. Robustness: add noisy copies (typos, abbreviations, loose identifiers) to part of
   the training set, and write a fully noisy copy of the test set.
5. Write the splits and a stats file.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from distill.data.example import Example, read_examples, write_examples
from distill.data.noise import add_noise
from distill.embeddings import Embedder, semantic_duplicates, semantic_leaks

NEAR_DUPLICATE_THRESHOLD = 0.85
LEAKAGE_THRESHOLD = 0.7
SEMANTIC_DUPLICATE_THRESHOLD = 0.95
SEMANTIC_LEAKAGE_THRESHOLD = 0.9


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def shingles(text: str, n: int = 3) -> frozenset[str]:
    words = normalize(text).split()
    if len(words) < n:
        return frozenset([" ".join(words)])
    return frozenset(" ".join(words[i : i + n]) for i in range(len(words) - n + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def deduplicate(examples: list[Example]) -> tuple[list[Example], int]:
    """Keep the first of each group of exact or near duplicates within a group."""
    kept: list[Example] = []
    seen_exact: set[str] = set()
    kept_shingles: dict[str, list[frozenset[str]]] = defaultdict(list)
    for example in examples:
        key = normalize(example.user_message)
        if key in seen_exact:
            continue
        sh = shingles(example.user_message)
        if any(
            jaccard(sh, other) >= NEAR_DUPLICATE_THRESHOLD for other in kept_shingles[example.group]
        ):
            continue
        seen_exact.add(key)
        kept_shingles[example.group].append(sh)
        kept.append(example)
    return kept, len(examples) - len(kept)


def deduplicate_semantic(
    examples: list[Example], embedder: Embedder, threshold: float = SEMANTIC_DUPLICATE_THRESHOLD
) -> tuple[list[Example], int]:
    drop = semantic_duplicates(
        [e.user_message for e in examples], [e.label_key for e in examples], embedder, threshold
    )
    return [e for i, e in enumerate(examples) if i not in drop], len(drop)


def stratified_split(
    examples: list[Example], test_ratio: float, seed: int
) -> tuple[list[Example], list[Example]]:
    by_group: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        by_group[example.group].append(example)
    rng = random.Random(seed)
    train: list[Example] = []
    test: list[Example] = []
    for group in sorted(by_group):
        items = sorted(by_group[group], key=lambda e: e.id)
        rng.shuffle(items)
        n_test = round(len(items) * test_ratio)
        if n_test == 0 and len(items) >= 4:
            n_test = 1
        test.extend(items[:n_test])
        train.extend(items[n_test:])
    return train, test


def remove_leakage(
    train: list[Example], test: list[Example]
) -> tuple[list[Example], list[Example], int]:
    train_shingles: dict[str, list[frozenset[str]]] = defaultdict(list)
    for example in train:
        train_shingles[example.group].append(shingles(example.user_message))
    clean_test: list[Example] = []
    moved = 0
    for example in test:
        sh = shingles(example.user_message)
        if any(jaccard(sh, other) >= LEAKAGE_THRESHOLD for other in train_shingles[example.group]):
            train.append(example)
            moved += 1
        else:
            clean_test.append(example)
    return train, clean_test, moved


def remove_semantic_leakage(
    train: list[Example],
    test: list[Example],
    embedder: Embedder,
    threshold: float = SEMANTIC_LEAKAGE_THRESHOLD,
) -> tuple[list[Example], list[Example], int]:
    leaks = semantic_leaks(
        [e.user_message for e in test],
        [e.label_key for e in test],
        [e.user_message for e in train],
        [e.label_key for e in train],
        embedder,
        threshold,
    )
    moved = [e for i, e in enumerate(test) if i in leaks]
    kept = [e for i, e in enumerate(test) if i not in leaks]
    return train + moved, kept, len(moved)


def noisy_copy(example: Example, seed: int, strength: float = 1.0) -> Example:
    rng = random.Random(f"{seed}-{example.id}")
    message = add_noise(example.user_message, example.language, rng, strength)
    return example.model_copy(
        update={"id": f"{example.id}~noisy", "user_message": message, "noisy": True}
    )


def add_noisy_copies(examples: list[Example], ratio: float, seed: int) -> list[Example]:
    """Noisy copies of a random share of the examples. Copies identical to their
    original (nothing was changed) are skipped."""
    copies = []
    for example in examples:
        if random.Random(f"{seed}-pick-{example.id}").random() < ratio:
            copy = noisy_copy(example, seed)
            if copy.user_message != example.user_message:
                copies.append(copy)
    return copies


def describe(examples: list[Example]) -> dict[str, dict[str, int]]:
    tools = Counter(call.name for e in examples for call in e.tool_calls)
    return {
        "category": dict(Counter(e.category for e in examples).most_common()),
        "language": dict(Counter(e.language for e in examples).most_common()),
        "tool": dict(tools.most_common()),
    }


def build(
    in_dir: Path,
    out_dir: Path,
    test_ratio: float = 0.15,
    seed: int = 42,
    noisy_ratio: float = 0.35,
    embedder: Embedder | None = None,
) -> dict:
    examples = read_examples(str(in_dir / "raw.jsonl"))
    examples.sort(key=lambda e: e.id)

    unique, duplicates = deduplicate(examples)
    semantic_dups = 0
    if embedder is not None:
        unique, semantic_dups = deduplicate_semantic(unique, embedder)

    train, test = stratified_split(unique, test_ratio, seed)
    train, test, moved = remove_leakage(train, test)
    semantic_moved = 0
    if embedder is not None:
        train, test, semantic_moved = remove_semantic_leakage(train, test, embedder)

    # Noise is added after the split, so a test example never has a copy in train.
    noisy_train = add_noisy_copies(train, noisy_ratio, seed)
    test_noisy = [noisy_copy(e, seed) for e in test]
    train = sorted(train + noisy_train, key=lambda e: e.id)
    test.sort(key=lambda e: e.id)
    test_noisy.sort(key=lambda e: e.id)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_examples(str(out_dir / "train.jsonl"), train)
    write_examples(str(out_dir / "test.jsonl"), test)
    write_examples(str(out_dir / "test_noisy.jsonl"), test_noisy)

    rejected: Counter[str] = Counter()
    rejected_path = in_dir / "rejected.jsonl"
    if rejected_path.exists():
        with rejected_path.open(encoding="utf-8") as f:
            rejected.update(json.loads(line)["reason"] for line in f if line.strip())

    stats = {
        "generated": len(examples) + sum(rejected.values()),
        "accepted": len(examples),
        "rejected": dict(rejected.most_common()),
        "duplicates_removed": {"lexical": duplicates, "semantic": semantic_dups},
        "moved_to_train_for_leakage": {"lexical": moved, "semantic": semantic_moved},
        "embeddings": getattr(embedder, "model_name", None) if embedder else None,
        "train": {"total": len(train), "noisy_copies": len(noisy_train), **describe(train)},
        "test": {"total": len(test), **describe(test)},
        "test_noisy": {"total": len(test_noisy)},
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate and split generated examples.")
    parser.add_argument("--in", dest="in_dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--noisy-ratio", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--embeddings", action="store_true", help="also deduplicate by meaning (needs [embeddings])"
    )
    args = parser.parse_args()

    embedder = None
    if args.embeddings:
        from distill.embeddings import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder()

    stats = build(args.in_dir, args.out, args.test_ratio, args.seed, args.noisy_ratio, embedder)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
