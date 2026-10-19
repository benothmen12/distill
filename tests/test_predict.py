import json
from datetime import datetime

import pytest

from distill.data.example import Example, ToolCall
from distill.eval.predict import OpenAIToolModel, parse_text_tool_calls

try:  # openai >= 3 ships its own HTTP library, older versions use httpx
    import httpx2 as httpx
except ImportError:
    import httpx

EXAMPLE = Example(
    id="e1",
    category="single",
    language="fr",
    now=datetime(2026, 10, 20, 9, 15),
    user_message="la M-104 vibre, statut ?",
    tool_calls=[ToolCall(name="get_machine_status", arguments={"machine_id": "M-104"})],
    source="test",
)


def _completion(message: dict) -> dict:
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {"index": 0, "message": {"role": "assistant", **message}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 900, "completion_tokens": 20, "total_tokens": 920},
    }


def _model(response: httpx.Response, requests: list) -> OpenAIToolModel:
    def handler(request):
        requests.append(json.loads(request.content))
        return response

    return OpenAIToolModel(
        model="m",
        api_key="k",
        base_url="https://api.example.test/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_structured_tool_calls_are_read():
    requests: list = []
    call = {
        "id": "c1",
        "type": "function",
        "function": {"name": "get_machine_status", "arguments": '{"machine_id": "M-104"}'},
    }
    model = _model(
        httpx.Response(200, json=_completion({"content": None, "tool_calls": [call]})), requests
    )
    prediction = model.predict(EXAMPLE)
    assert [c.name for c in prediction.tool_calls] == ["get_machine_status"]
    assert prediction.prompt_tokens == 900
    body = requests[0]
    assert body["temperature"] == 0
    assert len(body["tools"]) == 12
    assert [m["role"] for m in body["messages"]] == ["system", "user"]  # never the answer


def test_tool_calls_written_as_text_are_read():
    content = (
        "<think>The operator wants the status.</think>\n<tool_call>\n"
        '{"name": "get_machine_status", "arguments": {"machine_id": "M-104"}}\n</tool_call>'
    )
    model = _model(httpx.Response(200, json=_completion({"content": content})), [])
    prediction = model.predict(EXAMPLE)
    assert prediction.tool_calls[0].name == "get_machine_status"
    assert json.loads(prediction.tool_calls[0].arguments) == {"machine_id": "M-104"}
    assert prediction.content is None


def test_a_malformed_tool_call_is_scored_not_retried():
    failed = httpx.Response(
        400, json={"error": {"message": "Failed to call a function", "code": "tool_use_failed"}}
    )
    requests: list = []
    prediction = _model(failed, requests).predict(EXAMPLE)
    assert prediction.error.startswith("tool_use_failed")
    assert len(requests) == 1


def test_other_bad_requests_stop():
    import openai

    bad = httpx.Response(
        400, json={"error": {"message": "model not found", "code": "model_not_found"}}
    )
    with pytest.raises(openai.BadRequestError):
        _model(bad, []).predict(EXAMPLE)


def test_parse_text_tool_calls_keeps_the_rest_of_the_reply():
    calls, rest = parse_text_tool_calls("Which machine do you mean?")
    assert calls == [] and rest == "Which machine do you mean?"
    calls, _ = parse_text_tool_calls("<tool_call>{not json}</tool_call>")
    assert calls[0].name == "<unparsable>"
