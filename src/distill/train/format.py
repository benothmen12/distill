"""Turn examples into the exact text the model is trained on.

Each example becomes a prompt and a completion, rendered with the model's own
chat template and the 12 tool definitions:

- prompt: system prompt, tools and operator message, up to "assistant:";
- completion: the expected answer (tool calls or a reply) and the end token.

The training loss is computed on the completion only, so the model learns to
answer, not to repeat the operator. Rendering with the model's own template
means training and inference see exactly the same format.
"""

from __future__ import annotations

from typing import Any

from distill.data.example import Example
from distill.tools import to_openai_tools


class TemplateMismatch(ValueError):
    """The chat template renders the prompt differently inside the full dialog."""


def training_messages(example: Example) -> list[dict[str, Any]]:
    """Chat messages with tool-call arguments as objects, not JSON strings.

    Chat templates serialize arguments themselves; a string would be encoded
    twice by some of them ("{\\"machine_id\\": ...}").
    """
    messages = example.to_messages()
    answer = messages[-1]
    if answer.get("tool_calls"):
        messages[-1] = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in example.tool_calls
            ],
        }
    return messages


def render(example: Example, tokenizer: Any, tools: list[dict] | None = None) -> dict[str, str]:
    """Return {"prompt", "completion"} rendered with the tokenizer's chat template."""
    tools = tools if tools is not None else to_openai_tools()
    messages = training_messages(example)
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tools=tools, add_generation_prompt=True, tokenize=False
    )
    full = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)
    if not full.startswith(prompt):
        raise TemplateMismatch(
            f"{example.id}: the prompt is not a prefix of the full dialog; "
            "check the chat template before training."
        )
    completion = full[len(prompt) :].rstrip()
    if not completion:
        raise TemplateMismatch(f"{example.id}: empty completion")
    return {"prompt": prompt, "completion": completion}


def build_rows(examples: list[Example], tokenizer: Any) -> list[dict[str, str]]:
    tools = to_openai_tools()
    return [render(example, tokenizer, tools) for example in examples]


def length_report(rows: list[dict[str, str]], tokenizer: Any) -> dict[str, int]:
    """Token lengths, to check that max_length truncates nothing."""
    lengths = sorted(len(tokenizer(r["prompt"] + r["completion"])["input_ids"]) for r in rows)
    if not lengths:
        return {"n": 0}
    return {
        "n": len(lengths),
        "median": lengths[len(lengths) // 2],
        "p99": lengths[min(len(lengths) - 1, int(len(lengths) * 0.99))],
        "max": lengths[-1],
    }
