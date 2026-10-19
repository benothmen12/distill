"""Score one prediction against one expected answer.

An answer is correct only if it does exactly what the operator asked:

- tool calls: the right tools, each with every argument right. Arguments are
  compared after parsing with the tool schema, so "14:00" equals "14:00:00" and an
  omitted default equals the default written out. Parallel calls may come in any
  order. Free-text arguments (an issue, a message) are compared by meaning.
- clarification: no tool called, and a question asked.
- out of scope: no tool called, and a reply given.

Every wrong answer gets one outcome that says what went wrong, for error analysis.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from itertools import permutations

from distill.data.checks import fold
from distill.data.example import Example
from distill.data.specs import FREE_TEXT_FIELDS
from distill.eval.predict import Prediction
from distill.tools import InvalidToolCall, Tool, validate_call

TextSimilarity = Callable[[str, str], float]

LEXICAL_THRESHOLD = 0.5
SEMANTIC_THRESHOLD = 0.7

OUTCOMES = (
    "correct",
    "wrong_arguments",
    "wrong_tool",
    "missing_call",
    "extra_call",
    "invalid_call",
    "did_not_call",
    "called_instead_of_asking",
    "called_out_of_scope",
    "no_question",
    "empty_reply",
    "api_error",
)


def lexical_similarity(a: str, b: str) -> float:
    """F1 overlap of content words, matched on their first four letters."""

    def stems(text: str) -> Counter[str]:
        return Counter(w[:4] for w in fold(text).split() if len(w) >= 3)

    sa, sb = stems(a), stems(b)
    if not sa or not sb:
        return 1.0 if fold(a).strip() == fold(b).strip() else 0.0
    overlap = sum((sa & sb).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / sum(sb.values()), overlap / sum(sa.values())
    return 2 * precision * recall / (precision + recall)


@dataclass
class Score:
    id: str
    category: str
    language: str
    noisy: bool
    expected_tools: list[str]
    outcome: str
    fields: dict[str, bool] = field(default_factory=dict)  # "tool.argument" -> correct
    latency_ms: float | None = None

    @property
    def correct(self) -> bool:
        return self.outcome == "correct"

    @property
    def tool_correct(self) -> bool:
        """Right tools (or rightly no tool), whatever the arguments."""
        return self.outcome in ("correct", "wrong_arguments", "no_question", "empty_reply")


def _compare_calls(
    expected: list[Tool], predicted: list[Tool], similarity: TextSimilarity, threshold: float
) -> dict[str, bool]:
    fields: dict[str, bool] = {}
    for exp, pred in zip(expected, predicted, strict=True):
        for name in type(exp).model_fields:
            a, b = getattr(exp, name), getattr(pred, name)
            if name in FREE_TEXT_FIELDS and a is not None and b is not None:
                ok = similarity(str(a), str(b)) >= threshold
            elif isinstance(a, datetime) and isinstance(b, datetime):
                # Plant time is local: "14:00Z" and "14:00" mean the same moment here.
                ok = a.replace(tzinfo=None) == b.replace(tzinfo=None)
            else:
                ok = a == b
            key = f"{exp.name}.{name}"
            fields[key] = fields.get(key, True) and ok
    return fields


def _best_alignment(
    expected: list[Tool], predicted: list[Tool], similarity: TextSimilarity, threshold: float
) -> dict[str, bool]:
    """Pair predicted calls with expected ones in the order that matches best."""
    best: dict[str, bool] | None = None
    for order in permutations(predicted):
        if [c.name for c in order] != [c.name for c in expected]:
            continue
        fields = _compare_calls(expected, list(order), similarity, threshold)
        if best is None or sum(fields.values()) > sum(best.values()):
            best = fields
            if all(fields.values()):
                break
    return best or {}


def score_example(
    example: Example,
    prediction: Prediction,
    similarity: TextSimilarity = lexical_similarity,
    threshold: float = LEXICAL_THRESHOLD,
) -> Score:
    base = {
        "id": example.id,
        "category": example.category,
        "language": example.language,
        "noisy": example.noisy,
        "expected_tools": sorted(c.name for c in example.tool_calls),
        "latency_ms": prediction.latency_ms,
    }

    if prediction.error and not prediction.tool_calls:
        if prediction.error.startswith("tool_use_failed"):
            outcome = "invalid_call" if example.tool_calls else _wrong_call_outcome(example)
            return Score(**base, outcome=outcome)
        return Score(**base, outcome="api_error")

    parsed: list[Tool] = []
    valid = True
    for call in prediction.tool_calls:
        try:
            parsed.append(validate_call(call.name, call.arguments))
        except InvalidToolCall:
            valid = False

    if not example.tool_calls:
        if prediction.tool_calls:
            return Score(**base, outcome=_wrong_call_outcome(example))
        reply = (prediction.content or "").strip()
        if example.category == "clarify":
            return Score(**base, outcome="correct" if "?" in reply else "no_question")
        return Score(**base, outcome="correct" if reply else "empty_reply")

    if not prediction.tool_calls:
        return Score(**base, outcome="did_not_call")
    if not valid:
        return Score(**base, outcome="invalid_call")

    expected_names = Counter(c.name for c in example.tool_calls)
    predicted_names = Counter(c.name for c in parsed)
    if expected_names != predicted_names:
        if len(parsed) < len(example.tool_calls):
            outcome = "missing_call"
        elif len(parsed) > len(example.tool_calls):
            outcome = "extra_call"
        else:
            outcome = "wrong_tool"
        return Score(**base, outcome=outcome)

    expected = [validate_call(c.name, c.arguments) for c in example.tool_calls]
    fields = _best_alignment(expected, parsed, similarity, threshold)
    outcome = "correct" if all(fields.values()) else "wrong_arguments"
    return Score(**base, outcome=outcome, fields=fields)


def _wrong_call_outcome(example: Example) -> str:
    return "called_instead_of_asking" if example.category == "clarify" else "called_out_of_scope"
