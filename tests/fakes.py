"""A fake teacher that answers deterministically, so tests run offline."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Sequence
from typing import Any

import numpy as np

from distill.data.prompts import SPECS_MARKER


class FakeTeacher:
    """Writes a correct output for every spec, unless told to make a mistake."""

    model = "fake-teacher"

    def __init__(self, mistake: str | None = None) -> None:
        self.mistake = mistake
        self.calls = 0

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        self.calls += 1
        specs = json.loads(user.split(SPECS_MARKER, 1)[1])
        return {"examples": [self._answer(spec) for spec in specs]}

    def _answer(self, spec: dict[str, Any]) -> dict[str, Any]:
        if spec["category"] == "out_of_scope":
            return {
                "spec_id": spec["spec_id"],
                "user_message": f"Can you tell me about {spec['topic']}?",
                "free_text": [],
                "assistant_reply": "Sorry, I can only help with plant operations.",
            }
        if spec["category"] == "clarify":
            message = (
                f"{spec['calls'][0]['action']} Can you do that for me? "
                f"Asked {spec['now']}, {spec['style']}."
            )
            if self.mistake == "leak_missing":
                message += " on M-999 and L1 and TCK-00001 and P-0001"
            return {
                "spec_id": spec["spec_id"],
                "user_message": message,
                "free_text": [{} for _ in spec["calls"]],
                "assistant_reply": "Which one do you mean?",
            }

        parts = []
        for call in spec["calls"]:
            values = " ".join(f"{k} {str(v).replace('_', ' ')}" for k, v in call["fixed"].items())
            texts = " ".join(f"{name}: Generated {name} text" for name in call["free_text"])
            parts.append(f"please handle {values} {texts}".strip())
        message = " and ".join(parts)
        if self.mistake == "drop_identifiers":
            message = "please handle it"
        return {
            "spec_id": spec["spec_id"],
            "user_message": message,
            "free_text": [
                {name: f"Generated {name} text" for name in call["free_text"]}
                for call in spec["calls"]
            ],
            "assistant_reply": None,
        }


class FakeEmbedder:
    """Character trigram hashing: similar spellings get similar vectors.

    It only captures surface similarity, which is enough to test the plumbing
    offline. The real embedder captures meaning across languages.
    """

    model_name = "fake-embedder"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            clean = unicodedata.normalize("NFKD", text.lower())
            padded = f"  {clean}  "
            for i in range(len(padded) - 2):
                digest = hashlib.md5(padded[i : i + 3].encode()).digest()
                vectors[row, int.from_bytes(digest[:4], "little") % self.dim] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-9)
