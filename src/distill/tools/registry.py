"""Lookup, export and validation of tools.

`to_openai_tools()` produces the `tools` list sent to any OpenAI-compatible API
(Groq, vLLM, llama.cpp server...). `validate_call()` checks a predicted tool call
against the Pydantic schema, so training data and model outputs are held to the
same rules.
"""

from __future__ import annotations

import inspect
import json
from typing import Any

from pydantic import ValidationError

from distill.tools.schemas import ALL_TOOLS, Tool

TOOLS_BY_NAME: dict[str, type[Tool]] = {tool.name: tool for tool in ALL_TOOLS}


class InvalidToolCall(ValueError):
    """Raised when a tool call does not match any tool or fails validation."""


def _clean_schema(node: Any) -> Any:
    """Drop auto-generated `title` keys: they add tokens and carry no meaning.

    Only string titles are removed, so a real argument named "title" is kept.
    """
    if isinstance(node, dict):
        return {
            k: _clean_schema(v)
            for k, v in node.items()
            if not (k == "title" and isinstance(v, str))
        }
    if isinstance(node, list):
        return [_clean_schema(v) for v in node]
    return node


def to_openai_tool(tool: type[Tool]) -> dict[str, Any]:
    """Convert one tool to the OpenAI function-calling format."""
    parameters = _clean_schema(tool.model_json_schema())
    # Pydantic copies the docstring here too; it is already the function description.
    parameters.pop("description", None)
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": inspect.cleandoc(tool.__doc__ or "").replace("\n", " "),
            "parameters": parameters,
        },
    }


def to_openai_tools() -> list[dict[str, Any]]:
    """Export every tool, in a stable order."""
    return [to_openai_tool(tool) for tool in ALL_TOOLS]


def validate_call(name: str, arguments: dict[str, Any] | str) -> Tool:
    """Return the parsed tool call, or raise InvalidToolCall with the reason."""
    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        raise InvalidToolCall(f"unknown tool: {name!r}")

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise InvalidToolCall(f"arguments are not valid JSON: {exc.msg}") from exc

    if not isinstance(arguments, dict):
        raise InvalidToolCall("arguments must be a JSON object")

    try:
        return tool.model_validate(arguments)
    except ValidationError as exc:
        raise InvalidToolCall(f"{name}: {exc.error_count()} invalid argument(s)") from exc
