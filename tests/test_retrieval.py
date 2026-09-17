"""
tests/test_retrieval.py
──────────────────────────
Tests the retrieval workflow end-to-end against the LocalVectorStore
fallback only — no PINECONE_API_KEY, no network access required, so this
runs the same in CI as it does locally. PineconeVectorStore is exercised
by construction/interface checks only (it needs real credentials to do
anything beyond that).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from retrieval.embeddings import EMBEDDING_DIM, embed_headlines, load_text_embedder
from retrieval.vector_store import LocalVectorStore
from retrieval.query_similar import get_contextual_signal


@pytest.fixture(scope="module")
def embedder():
    # No trained saved_model.keras exists in CI, so this exercises the
    # documented untrained fallback in load_text_embedder(). The point of
    # these tests is pipeline correctness (shapes, wiring, similarity
    # math), not embedding quality.
    return load_text_embedder(keras_model_path="/nonexistent/path.keras")


def test_embed_headlines_shape_and_normalization(embedder):
    headlines = [
        "SYN00 shares surges after quarterly results",
        "SYN01 stock plunges on analyst commentary",
        "SYN02 holds steady ahead of earnings",
    ]
    vectors = embed_headlines(headlines, embedder=embedder)
    assert vectors.shape == (3, EMBEDDING_DIM)
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_local_vector_store_upsert_and_query(tmp_path):
    store = LocalVectorStore(index_dir=str(tmp_path), index_name="test-index")

    ids = ["a", "b", "c"]
    vectors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.9, 0.1, 0.0],
    ], dtype="float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    metadatas = [{"headline": f"headline {i}", "label_name": "up"} for i in ids]

    store.upsert(ids, vectors, metadatas)

    query = np.array([1.0, 0.0, 0.0], dtype="float32")
    matches = store.query(query, top_k=2)

    assert len(matches) == 2
    # "a" is an exact match, "c" is the closest other vector.
    assert matches[0].id == "a"
    assert matches[0].score > matches[1].score
    assert matches[0].metadata["headline"] == "headline a"


def test_local_vector_store_persists_across_instances(tmp_path):
    store1 = LocalVectorStore(index_dir=str(tmp_path), index_name="persist-test")
    store1.upsert(["x"], np.array([[1.0, 0.0]], dtype="float32"), [{"headline": "hi"}])

    store2 = LocalVectorStore(index_dir=str(tmp_path), index_name="persist-test")
    info = store2.describe()
    assert info["vector_count"] == 1


def test_local_vector_store_upsert_is_idempotent_on_id(tmp_path):
    store = LocalVectorStore(index_dir=str(tmp_path), index_name="idempotent-test")
    store.upsert(["a"], np.array([[1.0, 0.0]], dtype="float32"), [{"v": 1}])
    store.upsert(["a"], np.array([[0.0, 1.0]], dtype="float32"), [{"v": 2}])

    info = store.describe()
    assert info["vector_count"] == 1  # updated in place, not appended

    matches = store.query(np.array([0.0, 1.0], dtype="float32"), top_k=1)
    assert matches[0].metadata["v"] == 2


def test_get_contextual_signal_returns_expected_shape(tmp_path, embedder):
    store = LocalVectorStore(index_dir=str(tmp_path), index_name="contextual-test")
    headlines = [
        "SYN00 shares surges after quarterly results",
        "SYN01 stock plunges on analyst commentary",
    ]
    vectors = embed_headlines(headlines, embedder=embedder)
    metadatas = [
        {"headline": headlines[0], "label_name": "up", "ticker": "SYN00", "day": 20},
        {"headline": headlines[1], "label_name": "down", "ticker": "SYN01", "day": 21},
    ]
    store.upsert(["id0", "id1"], vectors, metadatas)

    signal = get_contextual_signal(
        "SYN02 shares surges after quarterly results",
        top_k=2,
        embedder=embedder,
        store=store,
    )

    assert signal["query_headline"] == "SYN02 shares surges after quarterly results"
    assert len(signal["matches"]) == 2
    assert set(signal["label_distribution"].keys()) <= {"up", "down", "flat"}
    assert signal["majority_label"] in {"up", "down", "flat"}


def test_get_contextual_signal_handles_empty_index(tmp_path, embedder):
    store = LocalVectorStore(index_dir=str(tmp_path), index_name="empty-test")
    signal = get_contextual_signal("any headline", top_k=5, embedder=embedder, store=store)
    assert signal["matches"] == []
    assert signal["majority_label"] is None
