"""Multilingual sentence embeddings.

Used where comparing meaning matters more than comparing characters:

- semantic deduplication: "M-104 vibre beaucoup" and "forte vibration sur la M-104"
  are the same example if they lead to the same calls;
- semantic leakage check between train and test;
- scoring free-text arguments in evaluation: "Abnormal vibration" and "Vibrations
  anormales" are the same issue, even though no word matches.

The default model is multilingual (50+ languages, French and English included) and
small enough to run on a laptop CPU. Install with: pip install -e ".[embeddings]"
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence
from typing import Protocol

import numpy as np

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return one L2-normalized vector per text, shape (len(texts), dim)."""
        ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SystemExit(
                'Embeddings need extra packages: pip install -e ".[embeddings]"'
            ) from exc
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return self._model.encode(
            list(texts),
            batch_size=64,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )


def cosine_similarity(embedder: Embedder, a: str, b: str) -> float:
    """Similarity of meaning between two texts, from -1 to 1 (usually 0 to 1)."""
    vectors = embedder.encode([a, b])
    return float(vectors[0] @ vectors[1])


def semantic_duplicates(
    texts: Sequence[str],
    keys: Sequence[Hashable],
    embedder: Embedder,
    threshold: float,
) -> set[int]:
    """Indices of texts that repeat an earlier text with the same key.

    Only texts with the same key (the same expected answer) are compared. Two
    messages that say the same thing about M-104 and M-105 are both kept: their
    answers differ, and the model must learn to tell them apart.
    """
    by_key: dict[Hashable, list[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        by_key[key].append(index)

    vectors = embedder.encode(list(texts)) if texts else None
    duplicates: set[int] = set()
    for indices in by_key.values():
        kept: list[int] = []
        for index in indices:
            if kept and float(np.max(vectors[kept] @ vectors[index])) >= threshold:
                duplicates.add(index)
            else:
                kept.append(index)
    return duplicates


def semantic_leaks(
    test_texts: Sequence[str],
    test_keys: Sequence[Hashable],
    train_texts: Sequence[str],
    train_keys: Sequence[Hashable],
    embedder: Embedder,
    threshold: float,
) -> set[int]:
    """Indices of test texts that nearly repeat a training text with the same key."""
    if not test_texts or not train_texts:
        return set()
    test_vectors = embedder.encode(list(test_texts))
    train_vectors = embedder.encode(list(train_texts))
    train_by_key: dict[Hashable, list[int]] = defaultdict(list)
    for index, key in enumerate(train_keys):
        train_by_key[key].append(index)

    leaks: set[int] = set()
    for index, key in enumerate(test_keys):
        candidates = train_by_key.get(key)
        if (
            candidates
            and float(np.max(train_vectors[candidates] @ test_vectors[index])) >= threshold
        ):
            leaks.add(index)
    return leaks
