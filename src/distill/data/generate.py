"""Turn sampled specs into examples with the teacher, then validate every one.

Usage:
    python -m distill.data.generate --n 300

Generation is resumable: examples already written (accepted or rejected) are
skipped, so you can stop at any time, or hit the daily quota, and run the same
command again tomorrow.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from distill.data.checks import (
    any_identifier_in_message,
    check_value,
    is_grounded,
    is_placeholder,
    raw_enum_in_message,
)
from distill.data.example import Example, ToolCall
from distill.data.prompts import TEACHER_SYSTEM_PROMPT, build_teacher_prompt
from distill.data.specs import IDENTIFIER_FIELDS, Spec, sample_specs
from distill.data.teacher import QuotaExhausted, Teacher, TeacherError
from distill.tools import TOOLS_BY_NAME, InvalidToolCall, validate_call

MAX_MESSAGE_CHARS = 600


class Rejected(Exception):
    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


def _check_message(message: Any) -> str:
    if not isinstance(message, str) or len(message.split()) < 2:
        raise Rejected("bad_message", "empty or too short")
    if len(message) > MAX_MESSAGE_CHARS:
        raise Rejected("bad_message", "too long")
    lowered = message.lower()
    for name in TOOLS_BY_NAME:
        if name in lowered:
            raise Rejected("leaks_tool_name", name)
    raw = raw_enum_in_message(message)
    if raw:
        raise Rejected("raw_enum_value", raw)
    return message.strip()


def _check_reply(reply: Any, must_be_question: bool) -> str:
    if not isinstance(reply, str) or not reply.strip():
        raise Rejected("bad_reply", "missing assistant_reply")
    if must_be_question and "?" not in reply:
        raise Rejected("bad_reply", "clarification is not a question")
    return reply.strip()


ACCENTED = set("éèêëàâçîïôùûüœ")
FRENCH_WORDS = {
    "le",
    "la",
    "les",
    "un",
    "une",
    "des",
    "pour",
    "est",
    "merci",
    "svp",
    "vous",
    "pouvez",
    "moi",
    "avez",
    "besoin",
    "faut",
    "il",
    "elle",
    "dans",
    "sur",
    "et",
}
ENGLISH_WORDS = {
    "the",
    "is",
    "for",
    "please",
    "and",
    "with",
    "can",
    "you",
    "need",
    "check",
    "what",
    "how",
    "my",
    "this",
    "to",
    "of",
    "it",
    "now",
}


def _check_language(message: str, language: str) -> None:
    """Reject a message written in the other language than its label.

    Accents are a strong French signal; otherwise a clear majority of common
    words decides. Short messages with no signal pass.
    """
    lowered = message.lower()
    words = set(re.findall(r"\w+", lowered))
    fr_hits = len(words & FRENCH_WORDS)
    en_hits = len(words & ENGLISH_WORDS)
    if language == "en" and (
        any(c in ACCENTED for c in lowered) or (fr_hits >= 2 and fr_hits > en_hits)
    ):
        raise Rejected("language_mismatch", "labelled en, written in French")
    if language == "fr" and en_hits >= 2 and en_hits > fr_hits:
        raise Rejected("language_mismatch", "labelled fr, written in English")


def _free_text_values(spec: Spec, free_text: Any) -> list[dict]:
    """One dict of free-text values per call, whatever shape the teacher used.

    The teacher answers either with one object per call (the shape we ask for),
    or with a flat list of strings in field order, or with a value under the
    wrong call index. All three are read here. Anything still missing is
    rejected later by the normal checks, and every value is grounded in the message.
    """
    calls = spec.calls
    per_call: list[dict] = [{} for _ in calls]
    if not isinstance(free_text, list):
        return per_call
    named: dict[str, str] = {}
    loose: list[str] = []
    for index, item in enumerate(free_text):
        if isinstance(item, dict):
            if index < len(calls):
                per_call[index].update(item)
            named.update({k: v for k, v in item.items() if isinstance(v, str)})
        elif isinstance(item, str):
            loose.append(item)
    slots = [
        (i, name)
        for i, call in enumerate(calls)
        for name in call.free_text
        if name not in per_call[i]
    ]
    for i, name in slots:
        if name in named:
            per_call[i][name] = named[name]
    slots = [(i, name) for i, name in slots if name not in per_call[i]]
    if loose and len(loose) == len(slots):
        for (i, name), value in zip(slots, loose, strict=True):
            per_call[i][name] = value
    return per_call


def _build_calls(spec: Spec, message: str, free_text: Any) -> list[ToolCall]:
    per_call = _free_text_values(spec, free_text)
    calls: list[ToolCall] = []
    for index, call in enumerate(spec.calls):
        values = per_call[index]
        arguments = dict(call.fixed)
        for name in call.free_text:
            value = values.get(name)
            if not isinstance(value, str) or not value.strip():
                raise Rejected("missing_free_text", f"{call.tool}.{name}")
            value = value.strip()
            if is_placeholder(name, value):
                raise Rejected("placeholder_free_text", f"{call.tool}.{name}={value}")
            # A free-text value must restate the message, never add to it.
            if not is_grounded(value, message):
                raise Rejected("ungrounded_free_text", f"{call.tool}.{name}={value}")
            arguments[name] = value
        try:
            parsed = validate_call(call.tool, arguments)
        except InvalidToolCall as exc:
            raise Rejected("invalid_call", str(exc)) from exc
        # Every fixed value in the label must be readable from the message:
        # otherwise the label teaches the model to invent it.
        for name, value in call.fixed.items():
            reason = check_value(name, value, message)
            if reason:
                raise Rejected(reason, f"{call.tool}.{name}={value}")
        calls.append(
            ToolCall(name=call.tool, arguments=parsed.model_dump(mode="json", exclude_unset=True))
        )
    return calls


def assemble_example(spec: Spec, output: Any, source: str) -> Example:
    """Validate the teacher's output for one spec. Raises Rejected with the reason."""
    if not isinstance(output, dict):
        raise Rejected("no_output")
    message = _check_message(output.get("user_message"))
    _check_language(message, spec.language)
    common = {
        "id": spec.id,
        "category": spec.category,
        "language": spec.language,
        "now": spec.now,
        "user_message": message,
        "source": source,
    }

    if spec.category in ("single", "parallel"):
        calls = _build_calls(spec, message, output.get("free_text"))
        return Example(**common, tool_calls=calls)

    if spec.category == "clarify":
        if spec.missing in IDENTIFIER_FIELDS and any_identifier_in_message(spec.missing, message):
            raise Rejected("missing_value_leaked", spec.missing or "")
        reply = _check_reply(output.get("assistant_reply"), must_be_question=True)
        return Example(
            **common,
            assistant_reply=reply,
            intended_tool=spec.calls[0].tool,
            missing=spec.missing,
        )

    reply = _check_reply(output.get("assistant_reply"), must_be_question=False)
    return Example(**common, assistant_reply=reply)


@dataclass
class GenerationReport:
    accepted: int = 0
    rejected: int = 0
    reasons: Counter[str] = field(default_factory=Counter)
    stopped_early: bool = False


def _done_ids(*paths: Path) -> set[str]:
    done: set[str] = set()
    for path in paths:
        if path.exists():
            with path.open(encoding="utf-8") as f:
                done.update(json.loads(line)["id"] for line in f if line.strip())
    return done


def _match_outputs(batch: list[Spec], data: Any) -> list[Any]:
    outputs = data.get("examples") if isinstance(data, dict) else None
    if not isinstance(outputs, list):
        return [None] * len(batch)
    by_id = {o.get("spec_id"): o for o in outputs if isinstance(o, dict)}
    matched = [by_id.get(spec.id) for spec in batch]
    # Fall back to order when the teacher rewrote the ids but kept the list intact.
    if any(m is None for m in matched) and len(outputs) == len(batch):
        matched = [m if m is not None else outputs[i] for i, m in enumerate(matched)]
    return matched


def generate(
    specs: list[Spec],
    teacher: Teacher,
    out_dir: Path,
    batch_size: int = 8,
    log: Any = print,
) -> GenerationReport:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path, rejected_path = out_dir / "raw.jsonl", out_dir / "rejected.jsonl"
    done = _done_ids(raw_path, rejected_path)
    todo = [spec for spec in specs if spec.id not in done]
    report = GenerationReport()
    source = f"teacher:{teacher.model}"
    log(f"{len(done)} specs already done, {len(todo)} to generate.")

    with (
        raw_path.open("a", encoding="utf-8") as raw,
        rejected_path.open("a", encoding="utf-8") as rej,
    ):
        for start in range(0, len(todo), batch_size):
            batch = todo[start : start + batch_size]
            try:
                data = teacher.complete_json(TEACHER_SYSTEM_PROMPT, build_teacher_prompt(batch))
                outputs = _match_outputs(batch, data)
            except QuotaExhausted as exc:
                log(f"Quota reached, stopping. Run the same command later to resume.\n  {exc}")
                report.stopped_early = True
                break
            except TeacherError as exc:
                outputs = [None] * len(batch)
                log(f"Teacher error on this batch: {exc}")

            for spec, output in zip(batch, outputs, strict=True):
                try:
                    example = assemble_example(spec, output, source)
                except Rejected as rejection:
                    report.rejected += 1
                    report.reasons[rejection.reason] += 1
                    record = {
                        "id": spec.id,
                        "reason": rejection.reason,
                        "detail": rejection.detail,
                        "spec": spec.model_dump(mode="json"),
                        "output": output,
                    }
                    rej.write(json.dumps(record, ensure_ascii=False) + "\n")
                else:
                    report.accepted += 1
                    raw.write(example.model_dump_json() + "\n")
            raw.flush()
            rej.flush()
            processed = min(start + batch_size, len(todo))
            log(f"[{processed}/{len(todo)}] accepted {report.accepted}, rejected {report.rejected}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic examples with the teacher.")
    parser.add_argument("--n", type=int, default=300, help="total number of specs to cover")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("data/generated"))
    args = parser.parse_args()

    from distill.data.teacher import OpenAICompatibleTeacher

    teacher = OpenAICompatibleTeacher.from_env()
    report = generate(sample_specs(args.n, args.seed), teacher, args.out, args.batch_size)

    print(f"\nAccepted: {report.accepted}   Rejected: {report.rejected}")
    for reason, count in report.reasons.most_common():
        print(f"  {reason}: {count}")
    print(f"Tokens used this run: {teacher.total_tokens:,}")


if __name__ == "__main__":
    main()
