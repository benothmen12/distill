import json
from datetime import datetime

import pytest

from distill.data.example import Example, ToolCall
from distill.eval.predict import PredictedCall, Prediction
from distill.eval.scoring import lexical_similarity, score_example

NOW = datetime(2026, 10, 20, 9, 15)


def example(category="single", calls=(), reply=None, **kw):
    return Example(
        id="e1",
        category=category,
        language="en",
        now=NOW,
        user_message="test",
        tool_calls=[ToolCall(name=n, arguments=a) for n, a in calls],
        assistant_reply=reply,
        source="test",
        **kw,
    )


def predict(*calls, content=None, error=None):
    return Prediction(
        id="e1",
        model="m",
        tool_calls=[
            PredictedCall(name=n, arguments=a if isinstance(a, str) else json.dumps(a))
            for n, a in calls
        ],
        content=content,
        error=error,
    )


SENSOR = ("get_sensor_readings", {"machine_id": "M-104", "sensor": "vibration"})
STOCK_A = ("get_spare_part_stock", {"part_number": "P-2231"})
STOCK_B = ("get_spare_part_stock", {"part_number": "P-4410"})


def outcome(ex, pred):
    return score_example(ex, pred).outcome


def test_a_default_written_out_equals_an_omitted_default():
    pred = predict(("get_sensor_readings", {**SENSOR[1], "window_minutes": 60}))
    assert outcome(example(calls=[SENSOR]), pred) == "correct"


@pytest.mark.parametrize(
    "start", ["2026-10-21T14:00", "2026-10-21T14:00:00", "2026-10-21T14:00:00Z"]
)
def test_datetime_formats_are_equivalent(start):
    expected = {
        "machine_id": "M-104",
        "start_time": "2026-10-21T14:00:00",
        "duration_minutes": 90,
        "reason": "Bearing replacement",
    }
    pred = predict(("schedule_downtime", {**expected, "start_time": start}))
    assert outcome(example(calls=[("schedule_downtime", expected)]), pred) == "correct"


def test_wrong_argument_is_reported_by_field():
    pred = predict(("get_sensor_readings", {**SENSOR[1], "sensor": "temperature"}))
    score = score_example(example(calls=[SENSOR]), pred)
    assert score.outcome == "wrong_arguments"
    assert score.fields["get_sensor_readings.sensor"] is False
    assert score.fields["get_sensor_readings.machine_id"] is True
    assert score.tool_correct


def test_wrong_tool():
    pred = predict(("get_machine_status", {"machine_id": "M-104"}))
    assert outcome(example(calls=[SENSOR]), pred) == "wrong_tool"


def test_parallel_calls_in_any_order():
    ex = example(category="parallel", calls=[STOCK_A, STOCK_B])
    assert outcome(ex, predict(STOCK_B, STOCK_A)) == "correct"
    assert outcome(ex, predict(STOCK_A)) == "missing_call"
    assert outcome(ex, predict(STOCK_A, STOCK_B, STOCK_B)) == "extra_call"
    assert outcome(ex, predict(STOCK_A, STOCK_A)) == "wrong_arguments"


@pytest.mark.parametrize(
    "arguments", [{"machine_id": "104", "sensor": "vibration"}, '{"machine_id": "M-104",', "[]"]
)
def test_invalid_calls(arguments):
    assert outcome(example(calls=[SENSOR]), predict(("get_sensor_readings", arguments))) == (
        "invalid_call"
    )


def test_unknown_tool_is_invalid():
    assert outcome(example(calls=[SENSOR]), predict(("read_sensor", SENSOR[1]))) == "invalid_call"


def test_answering_in_text_instead_of_calling():
    assert outcome(example(calls=[SENSOR]), predict(content="Sure, checking.")) == "did_not_call"


def test_clarification():
    ex = example(
        category="clarify",
        reply="Which machine?",
        intended_tool="get_machine_status",
        missing="machine_id",
    )
    assert outcome(ex, predict(content="Which machine do you mean?")) == "correct"
    assert outcome(ex, predict(content="I will check it.")) == "no_question"
    guess = predict(("get_machine_status", {"machine_id": "M-101"}))
    assert outcome(ex, guess) == "called_instead_of_asking"


def test_out_of_scope():
    ex = example(category="out_of_scope", reply="I can only help with plant operations.")
    assert outcome(ex, predict(content="Sorry, I can't help with recipes.")) == "correct"
    assert outcome(ex, predict(content="")) == "empty_reply"
    assert outcome(ex, predict(("notify_team", {"team": "quality", "message": "lasagna"}))) == (
        "called_out_of_scope"
    )


def test_free_text_is_compared_by_meaning_not_exact_words():
    expected = {
        "machine_id": "M-104",
        "issue": "Abnormal vibration on the spindle",
        "priority": "high",
        "team": "mechanical",
    }
    ex = example(calls=[("create_maintenance_ticket", expected)])
    close = predict(
        ("create_maintenance_ticket", {**expected, "issue": "abnormal vibrations spindle"})
    )
    other = predict(
        ("create_maintenance_ticket", {**expected, "issue": "Oil leak under the press"})
    )
    assert outcome(ex, close) == "correct"
    assert outcome(ex, other) == "wrong_arguments"


def test_an_invented_note_is_wrong():
    expected = {"ticket_id": "TCK-04431", "status": "resolved"}
    ex = example(calls=[("update_ticket_status", expected)])
    pred = predict(("update_ticket_status", {**expected, "note": "Replaced faulty sensor"}))
    assert outcome(ex, pred) == "wrong_arguments"


def test_api_errors():
    assert outcome(example(calls=[SENSOR]), predict(error="tool_use_failed: bad")) == "invalid_call"
    clarify = example(category="clarify", reply="Which?", intended_tool="x", missing="machine_id")
    assert outcome(clarify, predict(error="tool_use_failed: bad")) == "called_instead_of_asking"
    assert outcome(example(calls=[SENSOR]), predict(error="timeout")) == "api_error"


def test_lexical_similarity():
    assert lexical_similarity("Vibrations anormales", "vibration anormale") > 0.9
    assert lexical_similarity("Vibrations anormales", "fuite d'huile") == 0.0
