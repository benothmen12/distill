"""The copilot's system prompt, shared by data generation, training and evaluation.

Keeping one definition matters: a model trained with one prompt and evaluated with
another is not being measured on what it learned.
"""

from __future__ import annotations

from datetime import datetime

COPILOT_SYSTEM_PROMPT = """\
You are the operations copilot of a manufacturing plant.
Current date and time: {now}.
Use the available tools to act on the operator's request. When the request needs \
several independent actions, call several tools at once.
If a detail a tool requires is missing, for example which machine, ask one short \
question instead of guessing.
If the request has nothing to do with plant operations, say briefly that you cannot \
help with it.
Always reply in the operator's language."""


def format_now(now: datetime) -> str:
    """Human-readable timestamp with the weekday, so relative dates can be resolved."""
    return now.strftime("%A %Y-%m-%d %H:%M")


def copilot_system_prompt(now: datetime) -> str:
    return COPILOT_SYSTEM_PROMPT.format(now=format_now(now))
