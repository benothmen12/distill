"""Client for the teacher model, through any OpenAI-compatible API.

The default is gpt-oss-120b on Groq's free tier. Free tiers have per-minute and
per-day limits, so the client waits on short rate limits and stops cleanly on a
daily quota: generation is resumable, so the next run simply continues.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Protocol

DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
MAX_WAIT_SECONDS = 120


class TeacherError(RuntimeError):
    """The teacher returned something unusable for this batch."""


class QuotaExhausted(RuntimeError):
    """A long rate limit, usually the daily quota. Resume later."""


class Teacher(Protocol):
    model: str

    def complete_json(self, system: str, user: str) -> dict[str, Any]: ...


class OpenAICompatibleTeacher:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        reasoning_effort: str | None = "low",
        temperature: float = 0.9,
        max_completion_tokens: int = 4096,
        max_attempts: int = 6,
        http_client: Any = None,
    ) -> None:
        from openai import OpenAI  # imported here so tests never need network clients

        self.model = model
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.max_attempts = max_attempts
        self.total_tokens = 0
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
            timeout=120,
            http_client=http_client,
        )

    @classmethod
    def from_env(cls) -> OpenAICompatibleTeacher:
        from dotenv import load_dotenv

        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise SystemExit("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
        return cls(
            model=os.getenv("TEACHER_MODEL", DEFAULT_MODEL),
            api_key=api_key,
            base_url=os.getenv("TEACHER_BASE_URL", DEFAULT_BASE_URL),
            reasoning_effort=os.getenv("TEACHER_REASONING_EFFORT", "low") or None,
        )

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        import openai

        extra_body = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else None
        for attempt in range(self.max_attempts):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_format={"type": "json_object"},
                    temperature=self.temperature,
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

            if response.usage:
                self.total_tokens += response.usage.total_tokens
            content = response.choices[0].message.content or ""
            try:
                return json.loads(content)
            except json.JSONDecodeError as exc:
                raise TeacherError(f"teacher returned invalid JSON: {exc.msg}") from exc

        raise TeacherError(f"teacher unavailable after {self.max_attempts} attempts")


def _retry_after_seconds(exc: Any) -> float | None:
    response = getattr(exc, "response", None)
    value = response.headers.get("retry-after") if response is not None else None
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None
