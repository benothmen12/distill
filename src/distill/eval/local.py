"""Evaluate a model loaded in this process (on Kaggle's GPU), without a server.

The prompt is rendered with the model's own chat template and the tool
definitions, exactly as during training. Generation is greedy, in batches, and
the `<tool_call>` blocks the model writes are parsed into tool calls.
"""

from __future__ import annotations

import time
from typing import Any

from distill.data.example import Example
from distill.eval.predict import Prediction, parse_text_tool_calls
from distill.tools import to_openai_tools

END_MARKERS = ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>")


def clean_output(text: str) -> str:
    """Cut at the first end marker and drop padding."""
    for marker in END_MARKERS:
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip()


class LocalToolModel:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        name: str,
        batch_size: int = 8,
        max_new_tokens: int = 384,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.name = name
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self._tools = to_openai_tools()
        self.tokenizer.padding_side = "left"  # generation continues on the right
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _prompt(self, example: Example) -> str:
        return self.tokenizer.apply_chat_template(
            example.to_messages()[:2], tools=self._tools, add_generation_prompt=True, tokenize=False
        )

    def predict_batch(self, examples: list[Example]) -> list[Prediction]:
        import torch

        prompts = [self._prompt(e) for e in examples]
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self.model.device)
        start = time.perf_counter()
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        latency = (time.perf_counter() - start) * 1000 / len(examples)
        new_tokens = output[:, inputs["input_ids"].shape[1] :]
        texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=False)

        predictions = []
        for i, (example, text, ids) in enumerate(zip(examples, texts, new_tokens, strict=True)):
            calls, content = parse_text_tool_calls(clean_output(text))
            predictions.append(
                Prediction(
                    id=example.id,
                    model=self.name,
                    tool_calls=calls,
                    content=content,
                    latency_ms=round(latency, 1),
                    prompt_tokens=int(inputs["attention_mask"][i].sum()),
                    completion_tokens=int((ids != self.tokenizer.pad_token_id).sum()),
                )
            )
        return predictions

    def predict(self, example: Example) -> Prediction:
        return self.predict_batch([example])[0]
