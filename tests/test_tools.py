import pytest

from distill.tools import (
    ALL_TOOLS,
    TOOLS_BY_NAME,
    InvalidToolCall,
    to_openai_tools,
    validate_call,
)


def test_tool_names_are_unique():
    assert len(TOOLS_BY_NAME) == len(ALL_TOOLS)


def test_every_tool_has_a_description():
    for tool in to_openai_tools():
        assert len(tool["function"]["description"]) > 20


def test_openai_export_shape():
    for tool in to_openai_tools():
        assert tool["type"] == "function"
        params = tool["function"]["parameters"]
        assert params["type"] == "object"
        assert params.get("additionalProperties") is False
        assert "title" not in params
        assert "description" not in params  # already in function.description


def test_valid_call_is_parsed():
    call = validate_call(
        "create_maintenance_ticket",
        {
            "machine_id": "M-104",
            "issue": "Abnormal vibration",
            "priority": "high",
            "team": "mechanical",
        },
    )
    assert call.machine_id == "M-104"


def test_arguments_can_be_a_json_string():
    call = validate_call("get_failure_risk", '{"machine_id": "M-007", "horizon_hours": 48}')
    assert call.horizon_hours == 48


def test_defaults_are_applied():
    call = validate_call("get_sensor_readings", {"machine_id": "M-210", "sensor": "vibration"})
    assert call.window_minutes == 60


def test_datetime_without_seconds_is_accepted():
    call = validate_call(
        "schedule_downtime",
        {
            "machine_id": "M-104",
            "start_time": "2026-10-21T14:00",
            "duration_minutes": 90,
            "reason": "Bearing replacement",
        },
    )
    assert call.start_time.hour == 14


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("unknown_tool", {}),
        ("get_machine_status", {"machine_id": "104"}),  # wrong id format
        ("get_machine_status", {"machine_id": "M-104", "extra": 1}),  # unexpected argument
        ("get_sensor_readings", {"machine_id": "M-104", "sensor": "humidity"}),  # not in enum
        ("order_spare_part", {"part_number": "P-2231", "quantity": 0}),  # out of range
        ("get_failure_risk", "{not json"),  # malformed JSON
        ("get_failure_risk", "[1, 2]"),  # JSON but not an object
    ],
)
def test_invalid_calls_are_rejected(name, arguments):
    with pytest.raises(InvalidToolCall):
        validate_call(name, arguments)
