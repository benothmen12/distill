"""Aggregate scores into the metrics shown in the README."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import quantiles
from typing import Any

from distill.eval.scoring import Score


def _rate(hits: int, total: int) -> float | None:
    return round(hits / total, 4) if total else None


def _accuracy_by(scores: list[Score], key) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Score]] = defaultdict(list)
    for score in scores:
        for k in key(score):
            groups[k].append(score)
    return {
        k: {"n": len(v), "accuracy": _rate(sum(s.correct for s in v), len(v))}
        for k, v in sorted(groups.items())
    }


def summarize(scores: list[Score]) -> dict[str, Any]:
    calls = [s for s in scores if s.expected_tools]
    no_calls = [s for s in scores if not s.expected_tools]
    clarify = [s for s in no_calls if s.category == "clarify"]
    out_of_scope = [s for s in no_calls if s.category == "out_of_scope"]
    attempted = [s for s in calls if s.outcome not in ("did_not_call", "api_error")]

    field_hits: Counter[str] = Counter()
    field_totals: Counter[str] = Counter()
    for score in calls:
        for name, ok in score.fields.items():
            field_totals[name] += 1
            field_hits[name] += ok

    latencies = sorted(s.latency_ms for s in scores if s.latency_ms is not None)
    p50 = p95 = None
    if len(latencies) >= 2:
        cuts = quantiles(latencies, n=100, method="inclusive")
        p50, p95 = round(cuts[49], 1), round(cuts[94], 1)

    return {
        "n": len(scores),
        "accuracy": _rate(sum(s.correct for s in scores), len(scores)),
        "tool_calls": {
            "n": len(calls),
            "valid_call_rate": _rate(
                sum(s.outcome != "invalid_call" for s in attempted), len(attempted)
            ),
            "tool_accuracy": _rate(sum(s.tool_correct for s in calls), len(calls)),
            "argument_accuracy": _rate(
                sum(s.correct for s in calls), sum(s.tool_correct for s in calls)
            ),
            "accuracy": _rate(sum(s.correct for s in calls), len(calls)),
        },
        "abstention": {
            "n": len(no_calls),
            "no_call_rate": _rate(sum(s.tool_correct for s in no_calls), len(no_calls)),
            "clarification_accuracy": _rate(sum(s.correct for s in clarify), len(clarify)),
            "out_of_scope_accuracy": _rate(sum(s.correct for s in out_of_scope), len(out_of_scope)),
        },
        "latency_ms": {"p50": p50, "p95": p95},
        "outcomes": dict(Counter(s.outcome for s in scores).most_common()),
        "by_category": _accuracy_by(scores, lambda s: [s.category]),
        "by_language": _accuracy_by(scores, lambda s: [s.language]),
        "by_tool": _accuracy_by(calls, lambda s: sorted(set(s.expected_tools))),
        "by_field": {
            name: {"n": field_totals[name], "accuracy": _rate(field_hits[name], field_totals[name])}
            for name in sorted(field_totals)
        },
    }


def _pct(value: float | None) -> str:
    return "–" if value is None else f"{value * 100:.1f}%"


HEADLINE_ROWS = (
    ("Overall accuracy", ("accuracy",)),
    ("Valid call rate", ("tool_calls", "valid_call_rate")),
    ("Tool accuracy", ("tool_calls", "tool_accuracy")),
    ("Argument accuracy", ("tool_calls", "argument_accuracy")),
    ("Clarification accuracy", ("abstention", "clarification_accuracy")),
    ("Out-of-scope accuracy", ("abstention", "out_of_scope_accuracy")),
)


def _get(metrics: dict[str, Any], path: tuple[str, ...]) -> Any:
    for key in path:
        metrics = metrics.get(key) if isinstance(metrics, dict) else None
    return metrics


def headline_table(columns: dict[str, dict[str, Any]]) -> str:
    """Markdown table: one column per model or split, one row per headline metric."""
    names = list(columns)
    lines = [
        "| Metric | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for label, path in HEADLINE_ROWS:
        values = [_pct(_get(columns[n], path)) for n in names]
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    latency = [columns[n].get("latency_ms", {}).get("p50") for n in names]
    lines.append(
        "| Latency p50 | " + " | ".join("–" if v is None else f"{v:.0f} ms" for v in latency) + " |"
    )
    examples = [str(columns[n].get("n", "–")) for n in names]
    lines.append("| Examples | " + " | ".join(examples) + " |")
    return "\n".join(lines)
