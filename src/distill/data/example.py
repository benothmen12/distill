"""One training or test example, and its conversion to chat messages."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel

from distill.prompts import copilot_system_prompt


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]


class Example(BaseModel):
    id: str
    category: Literal["single", "parallel", "clarify", "out_of_scope"]
    language: Literal["en", "fr"]
    now: datetime
    user_message: str
    tool_calls: list[ToolCall] = []
    assistant_reply: str | None = None
    # Clarification examples: the action the operator wanted, and what they forgot.
    intended_tool: str | None = None
    missing: str | None = None
    # True for copies with injected typos, abbreviations and loose identifiers.
    noisy: bool = False
    source: str

    @property
    def expects_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def label_key(self) -> str:
        """Identical for two examples that expect exactly the same answer."""
        if self.tool_calls:
            calls = sorted(
                json.dumps([c.name, c.arguments], sort_keys=True, ensure_ascii=False)
                for c in self.tool_calls
            )
            return f"{self.category}|{'|'.join(calls)}"
        return f"{self.category}|{self.intended_tool}|{self.missing}"

    @property
    def group(self) -> str:
        """Stratification key: category plus the tools involved."""
        names = [call.name for call in self.tool_calls] or [self.intended_tool or "-"]
        tools = "+".join(sorted(names))
        return f"{self.category}:{tools}"

    def to_messages(self) -> list[dict[str, Any]]:
        """OpenAI chat format, as used for fine-tuning and evaluation."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": copilot_system_prompt(self.now)},
            {"role": "user", "content": self.user_message},
        ]
        if self.tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments, ensure_ascii=False),
                            },
                        }
                        for i, call in enumerate(self.tool_calls)
                    ],
                }
            )
        else:
            messages.append({"role": "assistant", "content": self.assistant_reply})
        return messages


def read_examples(path: str) -> list[Example]:
    with open(path, encoding="utf-8") as f:
        return [Example.model_validate_json(line) for line in f if line.strip()]


def write_examples(path: str, examples: list[Example]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(example.model_dump_json() + "\n")
