"""Ask a model to answer test examples, with the copilot's tools available.

Works with any OpenAI-compatible server: Groq for the teacher, vLLM or llama.cpp
for the student. The model sees exactly what it would see in production: the
copilot system prompt, the operator message and the 12 tool definitions.

Some servers return tool calls as text instead of structured `tool_calls`
(Qwen and Hermes models write `<tool_call>{...}</tool_call>`). Those are parsed
too, so a server setting does not change the score.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Protocol

from pydantic import BaseModel

from distill.data.example import Example
from distill.tools import to_openai_tools

MAX_WAIT_SECONDS = 120

TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


class PredictedCall(BaseModel):
    name: str
    arguments: str  # raw JSON text, exactly as the model produced it


class Prediction(BaseModel):
    id: str
    model: str
    tool_calls: list[PredictedCall] = []
    content: str | None = None
    error: str | None = None
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class QuotaExhausted(RuntimeError):
    """A long rate limit, usually the daily quota. Resume later."""


class ToolModel(Protocol):
    name: str

    def predict(self, example: Example) -> Prediction: ...


def parse_text_tool_calls(content: str | None) -> tuple[list[PredictedCall], str | None]:
    """Extract `<tool_call>` blocks written in plain text. Returns calls and the rest."""
    if not content:
        return [], content
    text = THINK_BLOCK.sub("", content)
    calls = []
    for block in TOOL_CALL_BLOCK.findall(text):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            calls.append(PredictedCall(name="<unparsable>", arguments=block))
            continue
        arguments = data.get("arguments", data.get("parameters", {}))
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        calls.append(PredictedCall(name=str(data.get("name", "")), arguments=arguments))
    rest = TOOL_CALL_BLOCK.sub("", text).strip() or None
    return calls, rest


class OpenAIToolModel:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        name: str | None = None,
        max_completion_tokens: int = 1024,
        reasoning_effort: str | None = None,
        max_attempts: int = 6,
        http_client: Any = None,
    ) -> None:
        from openai import OpenAI

        self.model = model
        self.name = name or model
        self.max_completion_tokens = max_completion_tokens
        self.reasoning_effort = reasoning_effort
        self.max_attempts = max_attempts
        self._tools = to_openai_tools()
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
            timeout=120,
            http_client=http_client,
        )

    def predict(self, example: Example) -> Prediction:
        import openai

        messages = example.to_messages()[:2]  # system + user; the answer is what we test
        extra_body = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else None
        for attempt in range(self.max_attempts):
            start = time.perf_counter()
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self._tools,
                    tool_choice="auto",
                    temperature=0,
                    max_completion_tokens=self.max_completion_tokens,
                    extra_body=extra_body,
                )
            except openai.RateLimitError as exc:
                wait = _retry_after_seconds(exc)
                if "per day" in str(exc).lower() or (wait or 0) > MAX_WAIT_SECONDS:
                    raise QuotaExhausted(str(exc)) from exc
                time.sleep(wait if wait is not None else min(2**attempt * 5, MAX_WAIT_SECONDS))
                continue
            except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError):
                time.sleep(min(2**attempt * 5, MAX_WAIT_SECONDS))
                continue
            except openai.BadRequestError as exc:
                # Groq rejects a tool call the model wrote incorrectly. That is the
                # model's mistake, so it is scored, not retried.
                if "tool_use_failed" in str(exc):
                    return Prediction(
                        id=example.id, model=self.name, error=f"tool_use_failed: {exc}"
                    )
                raise

            latency = (time.perf_counter() - start) * 1000
            message = response.choices[0].message
            calls = [
                PredictedCall(name=c.function.name, arguments=c.function.arguments or "{}")
                for c in (message.tool_calls or [])
            ]
            content = message.content
            if not calls:
                calls, content = parse_text_tool_calls(content)
            usage = response.usage
            return Prediction(
                id=example.id,
                model=self.name,
                tool_calls=calls,
                content=content,
                latency_ms=round(latency, 1),
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
            )

        return Prediction(id=example.id, model=self.name, error="unavailable after retries")


def _retry_after_seconds(exc: Any) -> float | None:
    response = getattr(exc, "response", None)
    value = response.headers.get("retry-after") if response is not None else None
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None
