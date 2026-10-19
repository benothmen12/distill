import json
from datetime import datetime

from distill.data.build import build, deduplicate, remove_leakage
from distill.data.example import Example, ToolCall
from distill.data.generate import generate
from distill.data.specs import sample_specs
from tests.fakes import FakeEmbedder, FakeTeacher


def _example(id_: str, message: str) -> Example:
    return Example(
        id=id_,
        category="single",
        language="en",
        now=datetime(2026, 10, 20, 9, 15),
        user_message=message,
        tool_calls=[ToolCall(name="get_machine_status", arguments={"machine_id": "M-104"})],
        source="test",
    )


def test_exact_and_near_duplicates_are_removed():
    examples = [
        _example("a", "What is the status of machine M-104 right now?"),
        _example("b", "what is the status of machine M-104 right now"),
        _example("c", "What is the status of machine M-104 right now please?"),
        _example("d", "Is M-104 running or stopped?"),
    ]
    kept, removed = deduplicate(examples)
    assert [e.id for e in kept] == ["a", "d"]
    assert removed == 2


def test_leaking_test_examples_move_to_train():
    train = [_example("a", "Give me the status of machine M-104 right now")]
    test = [
        _example("b", "Give me the status of machine M-104 right now thanks"),
        _example("c", "Is M-104 running?"),
    ]
    train, test, moved = remove_leakage(train, test)
    assert moved == 1
    assert [e.id for e in test] == ["c"]


def test_build_writes_disjoint_stratified_splits(tmp_path):
    generate(sample_specs(600), FakeTeacher(), tmp_path / "gen", log=lambda *a: None)
    stats = build(tmp_path / "gen", tmp_path / "out", test_ratio=0.2)

    def ids(name):
        with (tmp_path / "out" / name).open() as f:
            return {json.loads(line)["id"] for line in f}

    train, test = ids("train.jsonl"), ids("test.jsonl")
    assert train and test
    assert not train & test
    originals = {i for i in train if not i.endswith("~noisy")}
    noisy = train - originals
    removed = stats["duplicates_removed"]["lexical"]
    assert len(originals) + len(test) + removed == 600
    assert set(stats["test"]["category"]) == {"single", "parallel", "clarify", "out_of_scope"}
    assert (tmp_path / "out" / "stats.json").exists()

    # Noisy copies come only from training examples, never from test ones.
    assert noisy
    assert {i.removesuffix("~noisy") for i in noisy} <= originals
    assert ids("test_noisy.jsonl") == {f"{i}~noisy" for i in test}


def test_build_with_embeddings(tmp_path):
    generate(sample_specs(200), FakeTeacher(), tmp_path / "gen", log=lambda *a: None)
    stats = build(tmp_path / "gen", tmp_path / "out", embedder=FakeEmbedder())
    assert stats["embeddings"] == "fake-embedder"
    assert stats["duplicates_removed"]["semantic"] >= 0
    assert stats["test"]["total"] > 0
