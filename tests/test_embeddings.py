import numpy as np

from distill.embeddings import cosine_similarity, semantic_duplicates, semantic_leaks
from tests.fakes import FakeEmbedder


def test_vectors_are_normalized():
    vectors = FakeEmbedder().encode(["machine M-104 vibre", "check the stock of P-2231"])
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_similar_texts_score_higher():
    emb = FakeEmbedder()
    close = cosine_similarity(emb, "machine M-104 vibre beaucoup", "machine M104 vibre bcp")
    far = cosine_similarity(emb, "machine M-104 vibre beaucoup", "commande 3 roulements")
    assert close > far


def test_duplicates_are_only_dropped_when_the_answer_is_the_same():
    texts = [
        "status of machine M-104 please",
        "status of machine M-104 pls",  # same answer, nearly same text: duplicate
        "status of machine M-105 please",  # nearly same text, different answer: kept
    ]
    keys = ["status|M-104", "status|M-104", "status|M-105"]
    assert semantic_duplicates(texts, keys, FakeEmbedder(), threshold=0.7) == {1}


def test_leaks_need_the_same_answer():
    leaks = semantic_leaks(
        test_texts=["status of machine M-104 pls", "status of machine M-105 pls"],
        test_keys=["a", "b"],
        train_texts=["status of machine M-104 please"],
        train_keys=["a"],
        embedder=FakeEmbedder(),
        threshold=0.7,
    )
    assert leaks == {0}
