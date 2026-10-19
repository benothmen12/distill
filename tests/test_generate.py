import json

import pytest

from distill.data.example import read_examples
from distill.data.generate import Rejected, assemble_example, generate
from distill.data.specs import sample_specs
from distill.data.teacher import QuotaExhausted, TeacherError
from tests.fakes import FakeTeacher


def _quiet(*_args, **_kwargs):
    pass


def test_correct_outputs_are_all_accepted(tmp_path):
    specs = sample_specs(40)
    report = generate(specs, FakeTeacher(), tmp_path, batch_size=4, log=_quiet)
    assert report.accepted == 40
    assert report.rejected == 0

    examples = read_examples(str(tmp_path / "raw.jsonl"))
    assert {e.id for e in examples} == {s.id for s in specs}
    for example in examples:
        messages = example.to_messages()
        assert [m["role"] for m in messages] == ["system", "user", "assistant"]
        assert "Current date and time" in messages[0]["content"]
        if example.tool_calls:
            for call in messages[2]["tool_calls"]:
                json.loads(call["function"]["arguments"])
        else:
            assert messages[2]["content"]


def test_generation_resumes_where_it_stopped(tmp_path):
    specs = sample_specs(20)
    generate(specs[:10], FakeTeacher(), tmp_path, log=_quiet)
    teacher = FakeTeacher()
    report = generate(specs, teacher, tmp_path, batch_size=5, log=_quiet)
    assert report.accepted == 10
    assert teacher.calls == 2
    assert len(read_examples(str(tmp_path / "raw.jsonl"))) == 20


def test_missing_identifier_is_rejected(tmp_path):
    specs = [s for s in sample_specs(60) if s.category == "single"][:10]
    report = generate(specs, FakeTeacher(mistake="drop_identifiers"), tmp_path, log=_quiet)
    assert report.reasons["identifier_missing"] > 0
    assert report.accepted + report.rejected == 10


def test_clarification_that_leaks_the_missing_value_is_rejected(tmp_path):
    specs = [s for s in sample_specs(300) if s.category == "clarify"][:10]
    specs = [s for s in specs if s.missing in ("machine_id", "line_id", "ticket_id", "part_number")]
    assert specs
    report = generate(specs, FakeTeacher(mistake="leak_missing"), tmp_path, log=_quiet)
    assert report.reasons["missing_value_leaked"] == len(specs)


def test_tool_names_in_the_message_are_rejected():
    spec = next(s for s in sample_specs(50) if s.category == "out_of_scope")
    output = {"user_message": "call get_machine_status now", "assistant_reply": "No."}
    with pytest.raises(Rejected) as info:
        assemble_example(spec, output, "test")
    assert info.value.reason == "leaks_tool_name"


def test_teacher_error_rejects_the_batch_and_continues(tmp_path):
    class BrokenTeacher(FakeTeacher):
        def complete_json(self, system, user):
            raise TeacherError("bad JSON")

    report = generate(sample_specs(6), BrokenTeacher(), tmp_path, batch_size=3, log=_quiet)
    assert report.rejected == 6
    assert report.reasons["no_output"] == 6


def test_quota_stops_cleanly_without_losing_specs(tmp_path):
    class QuotaTeacher(FakeTeacher):
        def complete_json(self, system, user):
            if self.calls >= 1:
                raise QuotaExhausted("tokens per day")
            return super().complete_json(system, user)

    report = generate(sample_specs(9), QuotaTeacher(), tmp_path, batch_size=3, log=_quiet)
    assert report.stopped_early
    assert report.accepted == 3

    report = generate(sample_specs(9), FakeTeacher(), tmp_path, batch_size=3, log=_quiet)
    assert report.accepted == 6
