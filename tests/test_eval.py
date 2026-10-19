import json

from distill.data.build import build
from distill.data.example import read_examples
from distill.data.generate import generate
from distill.data.specs import sample_specs
from distill.eval.compare import collect
from distill.eval.report import headline_table
from distill.eval.run import base_id, evaluate, stratified_sample
from tests.fakes import FakeTeacher, OracleModel


def _quiet(*_a, **_k):
    pass


def _splits(tmp_path, n=400):
    generate(sample_specs(n), FakeTeacher(), tmp_path / "gen", log=_quiet)
    build(tmp_path / "gen", tmp_path / "data", test_ratio=0.25)
    test = read_examples(str(tmp_path / "data" / "test.jsonl"))
    noisy = read_examples(str(tmp_path / "data" / "test_noisy.jsonl"))
    return test, noisy


def test_stratified_sample_is_the_same_for_clean_and_noisy(tmp_path):
    test, noisy = _splits(tmp_path)
    clean_ids = {e.id for e in stratified_sample(test, 40)}
    noisy_ids = {base_id(e.id) for e in stratified_sample(noisy, 40)}
    assert len(clean_ids) == 40
    assert clean_ids == noisy_ids
    categories = {e.category for e in stratified_sample(test, 40)}
    assert categories == {"single", "parallel", "clarify", "out_of_scope"}


def test_a_perfect_model_scores_100_percent(tmp_path):
    test, _ = _splits(tmp_path)
    scores, metrics, finished = evaluate(test, OracleModel(), tmp_path / "res", log=_quiet)
    assert finished
    assert metrics["accuracy"] == 1.0
    assert metrics["tool_calls"]["valid_call_rate"] == 1.0
    assert metrics["abstention"]["clarification_accuracy"] == 1.0
    assert metrics["latency_ms"]["p50"] is not None
    assert all(v["accuracy"] == 1.0 for v in metrics["by_field"].values())
    saved = json.loads((tmp_path / "res" / "metrics.json").read_text())
    assert saved["complete"] is True


def test_evaluation_resumes_after_the_quota(tmp_path):
    test, _ = _splits(tmp_path)
    _, metrics, finished = evaluate(
        test[:10], OracleModel(fail_after=4), tmp_path / "r", log=_quiet
    )
    assert not finished
    assert metrics["n"] == 4
    assert metrics["complete"] is False

    model = OracleModel()
    _, metrics, finished = evaluate(test[:10], model, tmp_path / "r", log=_quiet)
    assert finished and model.calls == 6 and metrics["n"] == 10


def test_comparison_table(tmp_path):
    test, noisy = _splits(tmp_path)
    evaluate(test, OracleModel(), tmp_path / "results" / "oracle" / "test", log=_quiet)
    evaluate(noisy, OracleModel(), tmp_path / "results" / "oracle" / "test_noisy", log=_quiet)
    columns = collect([tmp_path / "results" / "oracle"])
    assert list(columns) == ["oracle", "oracle (noisy)"]
    table = headline_table(columns)
    assert "| Tool accuracy | 100.0% | 100.0% |" in table
    assert "Latency p50" in table
