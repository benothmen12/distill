"""Run a model on a test file, score it, and save the results.

Usage:
    # the teacher on Groq, on a stratified subset of 200 examples
    python -m distill.eval.run --name teacher --model openai/gpt-oss-120b --limit 200

    # the same subset, with typos and abbreviations
    python -m distill.eval.run --name teacher --model openai/gpt-oss-120b --limit 200 \
        --test data/processed/test_noisy.jsonl

    # a student served by vLLM or llama.cpp
    python -m distill.eval.run --name qwen3-4b-base --model Qwen/Qwen3-4B-Instruct-2507 \
        --base-url http://localhost:8000/v1 --api-key-env NONE

Predictions are saved as they come, so an interrupted run resumes where it
stopped. Results go to results/<name>/<test file>/.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from distill.data.example import Example, read_examples
from distill.eval.predict import Prediction, QuotaExhausted, ToolModel
from distill.eval.report import headline_table, summarize
from distill.eval.scoring import (
    LEXICAL_THRESHOLD,
    SEMANTIC_THRESHOLD,
    Score,
    lexical_similarity,
    score_example,
)

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"


def base_id(example_id: str) -> str:
    return example_id.removesuffix("~noisy")


def stratified_sample(examples: list[Example], limit: int | None, seed: int = 0) -> list[Example]:
    """A subset that keeps the share of each category and tool.

    Selection depends only on the clean id, so the clean and noisy test files
    give the same subset and their scores can be compared one to one.
    """
    if not limit or limit >= len(examples):
        return examples
    groups: dict[str, list[Example]] = defaultdict(list)
    for example in sorted(examples, key=lambda e: base_id(e.id)):
        groups[example.group].append(example)
    rng = random.Random(seed)
    for items in groups.values():
        rng.shuffle(items)

    total = len(examples)
    quotas = {g: limit * len(items) / total for g, items in groups.items()}
    counts = {g: int(q) for g, q in quotas.items()}
    by_remainder = sorted(groups, key=lambda g: (-(quotas[g] - counts[g]), g))
    for g in by_remainder[: limit - sum(counts.values())]:
        counts[g] += 1
    chosen = [e for g, items in groups.items() for e in items[: counts[g]]]
    return sorted(chosen, key=lambda e: e.id)


def _load_predictions(path: Path) -> dict[str, Prediction]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        predictions = [Prediction.model_validate_json(line) for line in f if line.strip()]
    return {p.id: p for p in predictions}


def evaluate(
    examples: list[Example],
    model: ToolModel,
    out_dir: Path,
    similarity=lexical_similarity,
    threshold: float = LEXICAL_THRESHOLD,
    log: Any = print,
) -> tuple[list[Score], dict[str, Any], bool]:
    """Predict (resuming), score and summarize. Returns scores, metrics, finished."""
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.jsonl"
    done = _load_predictions(predictions_path)
    todo = [e for e in examples if e.id not in done]
    log(f"{len(done)} predictions already saved, {len(todo)} to run.")

    finished = True
    # Local models generate in batches on the GPU; API models answer one at a time.
    batch_size = getattr(model, "batch_size", 1) if hasattr(model, "predict_batch") else 1
    with predictions_path.open("a", encoding="utf-8") as f:
        for start in range(0, len(todo), batch_size):
            batch = todo[start : start + batch_size]
            try:
                if batch_size > 1:
                    predictions = model.predict_batch(batch)
                else:
                    predictions = [model.predict(batch[0])]
            except QuotaExhausted as exc:
                log(f"Quota reached, stopping. Run the same command later to resume.\n  {exc}")
                finished = False
                break
            for prediction in predictions:
                f.write(prediction.model_dump_json() + "\n")
                done[prediction.id] = prediction
            f.flush()
            processed = start + len(batch)
            if processed % 20 < batch_size or processed == len(todo):
                log(f"[{processed}/{len(todo)}]")

    scores = [score_example(e, done[e.id], similarity, threshold) for e in examples if e.id in done]
    metrics = summarize(scores)
    metrics["model"] = model.name
    metrics["complete"] = finished and len(scores) == len(examples)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (out_dir / "scores.jsonl").open("w", encoding="utf-8") as f:
        for s in scores:
            f.write(json.dumps({**s.__dict__, "correct": s.correct}, ensure_ascii=False) + "\n")
    return scores, metrics, finished


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a model on the test set.")
    parser.add_argument("--name", required=True, help="results folder name, e.g. teacher")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--base-url", default=os.getenv("EVAL_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key-env", default="GROQ_API_KEY", help="env var with the key")
    parser.add_argument("--test", type=Path, default=Path("data/processed/test.jsonl"))
    parser.add_argument("--limit", type=int, default=None, help="stratified subset size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reasoning-effort", default=None, help="low/medium/high if supported")
    parser.add_argument("--embeddings", action="store_true", help="score free text by meaning")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    from dotenv import load_dotenv

    from distill.eval.predict import OpenAIToolModel

    load_dotenv()
    api_key = os.getenv(args.api_key_env, "") if args.api_key_env != "NONE" else "none"
    if not api_key:
        raise SystemExit(f"{args.api_key_env} is not set. Add it to your .env file.")
    model = OpenAIToolModel(
        model=args.model,
        api_key=api_key,
        base_url=args.base_url,
        name=args.name,
        reasoning_effort=args.reasoning_effort,
    )

    similarity, threshold = lexical_similarity, LEXICAL_THRESHOLD
    if args.embeddings:
        from distill.embeddings import SentenceTransformerEmbedder, cosine_similarity

        embedder = SentenceTransformerEmbedder()
        similarity = lambda a, b: cosine_similarity(embedder, a, b)  # noqa: E731
        threshold = SEMANTIC_THRESHOLD

    examples = stratified_sample(read_examples(str(args.test)), args.limit, args.seed)
    out_dir = args.out / args.name / args.test.stem
    _, metrics, finished = evaluate(examples, model, out_dir, similarity, threshold)

    print()
    print(headline_table({f"{args.name} ({args.test.stem})": metrics}))
    print(f"\nOutcomes: {metrics['outcomes']}")
    print(f"Saved to {out_dir}")
    if not finished:
        print("Not finished: run the same command again to continue.")


if __name__ == "__main__":
    main()
