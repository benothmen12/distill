"""Training format, tested with a tokenizer that renders like Qwen's chat template."""

import json
from datetime import datetime

import pytest

from distill.data.example import Example, ToolCall
from distill.eval.local import clean_output
from distill.eval.predict import Prediction, parse_text_tool_calls
from distill.eval.run import evaluate
from distill.eval.scoring import score_example
from distill.train.format import TemplateMismatch, build_rows, render, training_messages

NOW = datetime(2026, 10, 20, 9, 15)


class QwenLikeTokenizer:
    """Renders messages the way Qwen's template does, tools and <tool_call> included."""

    eos_token = "<|im_end|>"

    def apply_chat_template(
        self, messages, tools=None, add_generation_prompt=False, tokenize=False
    ):
        out = ""
        for m in messages:
            if m["role"] == "system":
                out += "<|im_start|>system\n" + m["content"]
                if tools:
                    out += "\n\n# Tools\n<tools>\n" + "\n".join(json.dumps(t) for t in tools)
                    out += "\n</tools>"
                out += "<|im_end|>\n"
            elif m["role"] == "user":
                out += f"<|im_start|>user\n{m['content']}<|im_end|>\n"
            else:
                out += "<|im_start|>assistant\n" + (m.get("content") or "")
                for i, call in enumerate(m.get("tool_calls") or []):
                    fn = call["function"]
                    args = fn["arguments"]
                    args = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
                    sep = "\n" if (i or m.get("content")) else ""
                    body = f'{{"name": "{fn["name"]}", "arguments": {args}}}'
                    out += f"{sep}<tool_call>\n{body}\n</tool_call>"
                out += "<|im_end|>\n"
        if add_generation_prompt:
            out += "<|im_start|>assistant\n"
        return out

    def __call__(self, text):
        return {"input_ids": text.split()}


class InconsistentTokenizer(QwenLikeTokenizer):
    """A template that renders the prompt differently when an answer follows."""

    def apply_chat_template(self, messages, **kwargs):
        text = super().apply_chat_template(messages, **kwargs)
        return text if kwargs.get("add_generation_prompt") else "DATE\n" + text


def _example(category="parallel", calls=None, reply=None):
    calls = calls or [
        ToolCall(
            name="create_maintenance_ticket",
            arguments={
                "machine_id": "M-104",
                "issue": "Vibrations anormales",
                "priority": "high",
                "team": "mechanical",
            },
        ),
        ToolCall(name="notify_team", arguments={"team": "mechanical", "message": "M-104 vibre"}),
    ]
    return Example(
        id="e1",
        category=category,
        language="fr",
        now=NOW,
        user_message="la M-104 vibre, ticket urgent et préviens les mécanos",
        tool_calls=calls if category in ("single", "parallel") else [],
        assistant_reply=reply,
        source="test",
    )


def test_tool_call_arguments_are_objects_not_strings():
    messages = training_messages(_example())
    args = messages[-1]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, dict) and args["machine_id"] == "M-104"


def test_prompt_and_completion_split():
    row = render(_example(), QwenLikeTokenizer())
    assert row["prompt"].endswith("<|im_start|>assistant\n")
    assert "la M-104 vibre" in row["prompt"]
    assert "<tools>" in row["prompt"] and row["prompt"].count('"type": "function"') == 12
    assert row["completion"].startswith("<tool_call>")
    assert row["completion"].endswith("<|im_end|>")
    assert "la M-104 vibre" not in row["completion"]  # the operator is never a target


def test_what_the_model_learns_is_read_back_exactly_by_the_exam():
    example = _example()
    completion = render(example, QwenLikeTokenizer())["completion"]
    calls, rest = parse_text_tool_calls(clean_output(completion))
    prediction = Prediction(id="e1", model="m", tool_calls=calls, content=rest)
    assert score_example(example, prediction).outcome == "correct"


def test_replies_are_trained_too():
    clarify = _example(category="clarify", reply="Quelle machine ?")
    row = render(clarify, QwenLikeTokenizer())
    assert row["completion"] == "Quelle machine ?<|im_end|>"


def test_inconsistent_template_is_caught_before_training():
    with pytest.raises(TemplateMismatch):
        render(_example(), InconsistentTokenizer())


def test_build_rows():
    rows = build_rows(
        [_example(), _example(category="out_of_scope", reply="Désolé.")], QwenLikeTokenizer()
    )
    assert len(rows) == 2 and all(set(r) == {"prompt", "completion"} for r in rows)


def test_clean_output_cuts_at_the_end_token():
    assert (
        clean_output("Quelle machine ?<|im_end|><|endoftext|><|endoftext|>") == "Quelle machine ?"
    )


def test_batched_models_are_evaluated_in_batches(tmp_path):
    class BatchOracle:
        name = "batch-oracle"
        batch_size = 3

        def __init__(self):
            self.batches = []

        def predict_batch(self, examples):
            self.batches.append(len(examples))
            return [
                Prediction(id=e.id, model=self.name, content="Quelle machine ?") for e in examples
            ]

    examples = [
        _example(category="clarify", reply="?").model_copy(update={"id": f"e{i}"}) for i in range(7)
    ]
    model = BatchOracle()
    scores, metrics, finished = evaluate(examples, model, tmp_path, log=lambda *a: None)
    assert finished and model.batches == [3, 3, 1]
    assert metrics["accuracy"] == 1.0
