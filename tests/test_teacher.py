import json

import pytest

try:  # openai >= 3 ships its own HTTP library, older versions use httpx
    import httpx2 as httpx
except ImportError:
    import httpx

from distill.data import teacher as teacher_module
from distill.data.teacher import OpenAICompatibleTeacher, QuotaExhausted, TeacherError


def _completion(content: str) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _teacher(responses: list[httpx.Response], requests: list[dict]) -> OpenAICompatibleTeacher:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return responses.pop(0)

    return OpenAICompatibleTeacher(
        model="test-model",
        api_key="test-key",
        base_url="https://api.example.test/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(teacher_module.time, "sleep", lambda _seconds: None)


def test_request_asks_for_json_with_low_reasoning():
    requests: list[dict] = []
    teacher = _teacher([httpx.Response(200, json=_completion('{"examples": []}'))], requests)
    assert teacher.complete_json("system", "user") == {"examples": []}
    body = requests[0]
    assert body["response_format"] == {"type": "json_object"}
    assert body["reasoning_effort"] == "low"
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert teacher.total_tokens == 15


def test_short_rate_limit_is_retried():
    requests: list[dict] = []
    limited = httpx.Response(
        429,
        headers={"retry-after": "2"},
        json={"error": {"message": "Rate limit reached: requests per minute"}},
    )
    teacher = _teacher([limited, httpx.Response(200, json=_completion("{}"))], requests)
    assert teacher.complete_json("s", "u") == {}
    assert len(requests) == 2


def test_daily_quota_stops_generation():
    requests: list[dict] = []
    daily = httpx.Response(
        429,
        headers={"retry-after": "30"},
        json={"error": {"message": "Rate limit reached on tokens per day (TPD)"}},
    )
    teacher = _teacher([daily], requests)
    with pytest.raises(QuotaExhausted):
        teacher.complete_json("s", "u")


def _json_validate_failed() -> httpx.Response:
    # The exact error Groq returned in the first full run.
    return httpx.Response(
        400,
        json={
            "error": {
                "message": "Failed to generate JSON. Please adjust your prompt.",
                "type": "invalid_request_error",
                "code": "json_validate_failed",
                "failed_generation": '{"examples":[{"free_text":[{},{}},"x":1}]}',
            }
        },
    )


def test_broken_json_is_asked_again():
    requests: list[dict] = []
    responses = [_json_validate_failed(), httpx.Response(200, json=_completion('{"ok": 1}'))]
    teacher = _teacher(responses, requests)
    assert teacher.complete_json("s", "u") == {"ok": 1}
    assert len(requests) == 2


def test_repeated_broken_json_gives_up_on_the_batch_only():
    requests: list[dict] = []
    teacher = _teacher([_json_validate_failed() for _ in range(3)], requests)
    with pytest.raises(TeacherError):
        teacher.complete_json("s", "u")
    assert len(requests) == 3


def test_other_bad_requests_stop_everything():
    import openai

    requests: list[dict] = []
    wrong_model = httpx.Response(
        400, json={"error": {"message": "model not found", "code": "model_not_found"}}
    )
    teacher = _teacher([wrong_model], requests)
    with pytest.raises(openai.BadRequestError):
        teacher.complete_json("s", "u")


def test_invalid_json_content_is_retried_then_a_teacher_error():
    requests: list[dict] = []
    teacher = _teacher(
        [httpx.Response(200, json=_completion("not json")) for _ in range(3)], requests
    )
    with pytest.raises(TeacherError):
        teacher.complete_json("s", "u")
    assert len(requests) == 3
