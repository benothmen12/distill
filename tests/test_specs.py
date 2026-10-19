from collections import Counter

from distill.data.specs import IDENTIFIER_FIELDS, sample_specs
from distill.tools import ALL_TOOLS, TOOLS_BY_NAME, validate_call


def test_specs_are_deterministic_and_prefix_stable():
    assert sample_specs(50, seed=7) == sample_specs(100, seed=7)[:50]


def test_different_seeds_give_different_specs():
    assert sample_specs(20, seed=1) != sample_specs(20, seed=2)


def test_sampled_calls_are_valid_once_free_text_is_filled():
    for spec in sample_specs(600):
        if spec.category not in ("single", "parallel"):
            continue
        for call in spec.calls:
            arguments = {**call.fixed, **{name: "Placeholder text" for name in call.free_text}}
            validate_call(call.tool, arguments)


def test_distribution_covers_everything():
    specs = sample_specs(1500)
    categories = Counter(s.category for s in specs)
    assert set(categories) == {"single", "parallel", "clarify", "out_of_scope"}
    assert {s.language for s in specs} == {"en", "fr"}
    tools = {call.tool for s in specs for call in s.calls}
    assert tools == {tool.name for tool in ALL_TOOLS}


def test_clarify_removes_one_required_argument():
    for spec in sample_specs(800):
        if spec.category != "clarify":
            continue
        (call,) = spec.calls
        assert spec.missing not in call.fixed
        assert TOOLS_BY_NAME[call.tool].model_fields[spec.missing].is_required()


def test_parallel_calls_are_distinct():
    for spec in sample_specs(800):
        if spec.category == "parallel":
            first, second = spec.calls
            assert (first.tool, first.fixed) != (second.tool, second.fixed)


def test_paired_calls_share_the_same_machine():
    for spec in sample_specs(800):
        if spec.category != "parallel":
            continue
        first, second = spec.calls
        if first.tool != second.tool:
            for field in IDENTIFIER_FIELDS:
                if field in first.fixed and field in second.fixed:
                    assert first.fixed[field] == second.fixed[field]
