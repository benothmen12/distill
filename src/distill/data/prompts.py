"""Prompts sent to the teacher to turn sampled specs into operator messages."""

from __future__ import annotations

import json
from typing import Any

from distill.data.specs import Spec
from distill.prompts import format_now
from distill.tools import TOOLS_BY_NAME, to_openai_tool

SPECS_MARKER = "SPECS:"

TEACHER_SYSTEM_PROMPT = """\
You write realistic training data for an AI copilot used in a manufacturing plant.
Operators (technicians, shift supervisors, maintenance managers, quality engineers) \
send it short chat messages. For each spec you receive, write the operator's message \
and fill the requested free-text fields. Return only JSON."""

RULES = """\
Rules:
1. The message must let a careful reader recover every value in "fixed" exactly:
   - identifiers as given (M-104, L3, TCK-00412, P-2231). In messy styles they may \
be written loosely ("M104", "machine 104", "ligne 3", "ticket 412"), but the number \
must always be there;
   - enum values expressed naturally but without ambiguity (priority "high" -> \
"high priority" / "priorité haute"; team "electrical" -> "the electrical team" / \
"l'équipe électrique"; kpi "oee" -> "OEE" / "TRS"). Never write an enum in its raw \
technical form: "in progress" / "en cours", never "in_progress";
   - every number in "fixed" must appear in the message, in digits or words; another \
unit is fine if the conversion is exact (120 minutes -> "2 hours");
   - dates and times absolute or relative to "now" ("tomorrow at 2 pm"), as long as \
they resolve exactly to the given value;
   - booleans: urgent=true means the operator asks for express delivery, false means \
they say nothing about urgency or ask for normal delivery.
2. Do not mention any argument that is not in "fixed" or "free_text": the copilot \
will use the default.
3. For each name in "free_text", write a short realistic value in the language of the \
message. It must restate what the operator actually says in the message, with the \
message's own words where possible: never add a fact the message does not contain, and \
never use the field name as the value (a keyword is a real word like "bearing").
4. Write in the spec's language ("en" English, "fr" French) and style. Vary the \
wording; never name the tools or use technical function names.
5. Category "parallel": one message that asks for all the calls.
6. Category "clarify": the operator wants the action described, but the message must \
not contain the "missing" argument or any hint of it. Also write "assistant_reply": \
one short question asking for it, in the message language.
7. Category "out_of_scope": a request about "topic", unrelated to plant operations. \
Also write "assistant_reply": one polite sentence saying it is outside what the \
copilot can help with, in the message language.

Return this JSON object, with one entry per spec, in the same order:
{"examples": [{"spec_id": "...", "user_message": "...", "free_text": [{...}, ...], \
"assistant_reply": "..." or null}]}
"free_text" has one object per call (an empty object when a call has no free text)."""


def _field_help(tool: str, names: list[str]) -> dict[str, str]:
    """What each argument means, so the teacher writes sensible values."""
    fields = TOOLS_BY_NAME[tool].model_fields
    return {name: fields[name].description or "" for name in names}


def _spec_for_teacher(spec: Spec) -> dict[str, Any]:
    return {
        "spec_id": spec.id,
        "category": spec.category,
        "language": spec.language,
        "style": spec.style,
        "now": format_now(spec.now),
        "calls": [
            {
                "action": to_openai_tool(TOOLS_BY_NAME[call.tool])["function"]["description"],
                "fixed": call.fixed,
                "free_text": call.free_text,
                "fields": _field_help(call.tool, [*call.fixed, *call.free_text]),
            }
            for call in spec.calls
        ],
        "missing": spec.missing,
        "topic": spec.topic,
    }


def build_teacher_prompt(specs: list[Spec]) -> str:
    # Compact JSON: indentation would cost about a third more tokens on a free quota.
    payload = json.dumps(
        [_spec_for_teacher(s) for s in specs], ensure_ascii=False, separators=(",", ":")
    )
    return f"{RULES}\n\n{SPECS_MARKER}\n{payload}"
