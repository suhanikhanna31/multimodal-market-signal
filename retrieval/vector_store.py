"""
retrieval/vector_store.py
────────────────────────────
A small `VectorStore` interface with two implementations:

  - `PineconeVectorStore` — real semantic retrieval backed by Pinecone,
    used when `PINECONE_API_KEY` is set.
  - `LocalVectorStore`    — a pure-numpy, file-persisted fallback with the
    exact same interface, used otherwise. This is what makes the retrieval
    workflow runnable in local development / CI / this sandbox without a
    Pinecone account or network access — no code changes needed to switch
    between the two, only environment variables.

Both stores speak the same three operations: `upsert`, `query`, `describe`.
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

DEFAULT_LOCAL_INDEX_DIR = os.path.join(os.path.dirname(__file__), "local_index")
DEFAULT_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "market-signal-headlines")


@dataclass
class Match:
    id: str
    score: float
    metadata: dict = field(default_factory=dict)


class VectorStore:
    """Common interface. Not meant to be instantiated directly."""

    def upsert(self, ids: List[str], vectors: np.ndarray, metadatas: List[dict]) -> None:
        raise NotImplementedError

    def query(self, vector: np.ndarray, top_k: int = 5) -> List[Match]:
        raise NotImplementedError

    def describe(self) -> dict:
        raise NotImplementedError


class LocalVectorStore(VectorStore):
    """
    Brute-force cosine-similarity search over vectors kept in memory and
    persisted to a local JSON + .npy file pair. Fine at this project's
    scale (a few thousand headlines); not meant to replace a real ANN
    index at production scale.
    """

    def __init__(self, index_dir: str = DEFAULT_LOCAL_INDEX_DIR, index_name: str = DEFAULT_INDEX_NAME):
        self.index_dir = index_dir
        self.index_name = index_name
        os.makedirs(self.index_dir, exist_ok=True)
        self._vectors_path = os.path.join(self.index_dir, f"{index_name}.vectors.npy")
        self._meta_path = os.path.join(self.index_dir, f"{index_name}.meta.json")
        self._ids: List[str] = []
        self._vectors: Optional[np.ndarray] = None
        self._metadatas: List[dict] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self._vectors_path) and os.path.exists(self._meta_path):
            self._vectors = np.load(self._vectors_path)
            with open(self._meta_path, "r") as f:
                payload = json.load(f)
            self._ids = payload["ids"]
            self._metadatas = payload["metadatas"]

    def _save(self) -> None:
        np.save(self._vectors_path, self._vectors)
        with open(self._meta_path, "w") as f:
            json.dump({"ids": self._ids, "metadatas": self._metadatas}, f)

    def upsert(self, ids: List[str], vectors: np.ndarray, metadatas: List[dict]) -> None:
        vectors = np.asarray(vectors, dtype="float32")
        existing = {id_: i for i, id_ in enumerate(self._ids)}

        if self._vectors is None:
            self._vectors = np.zeros((0, vectors.shape[1]), dtype="float32")

        for i, id_ in enumerate(ids):
            if id_ in existing:
                self._vectors[existing[id_]] = vectors[i]
                self._metadatas[existing[id_]] = metadatas[i]
            else:
                self._ids.append(id_)
                self._metadatas.append(metadatas[i])
                self._vectors = np.vstack([self._vectors, vectors[i][np.newaxis, :]])

        self._save()

    def query(self, vector: np.ndarray, top_k: int = 5) -> List[Match]:
        if self._vectors is None or len(self._ids) == 0:
            return []
        vector = np.asarray(vector, dtype="float32")
        # Vectors are assumed pre-normalized (see retrieval/embeddings.py),
        # so cosine similarity is just the dot product.
        scores = self._vectors @ vector
        top_k = min(top_k, len(scores))
        top_idx = np.argsort(-scores)[:top_k]
        return [
            Match(id=self._ids[i], score=float(scores[i]), metadata=self._metadatas[i])
            for i in top_idx
        ]

    def describe(self) -> dict:
        return {
            "backend": "local",
            "index_name": self.index_name,
            "vector_count": len(self._ids),
            "storage_path": self.index_dir,
        }


class PineconeVectorStore(VectorStore):
    """
    Thin wrapper around the Pinecone Python SDK (v3+ `pinecone` package).
    Creates the index (serverless, cosine metric) on first use if it
    doesn't already exist.
    """

    def __init__(
        self,
        index_name: str = DEFAULT_INDEX_NAME,
        dimension: int = 32,
        cloud: str = None,
        region: str = None,
    ):
        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError as e:
            raise ImportError(
                "The `pinecone` package is required for PineconeVectorStore. "
                "Install it with `pip install pinecone` (it's in requirements.txt)."
            ) from e

        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY is not set.")

        self.index_name = index_name
        self._pc = Pinecone(api_key=api_key)

        existing = [idx["name"] for idx in self._pc.list_indexes()]
        if index_name not in existing:
            self._pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=cloud or os.environ.get("PINECONE_CLOUD", "aws"),
                    region=region or os.environ.get("PINECONE_REGION", "us-east-1"),
                ),
            )
        self._index = self._pc.Index(index_name)

    def upsert(self, ids: List[str], vectors: np.ndarray, metadatas: List[dict]) -> None:
        vectors = np.asarray(vectors, dtype="float32")
        items = [
            {"id": id_, "values": vectors[i].tolist(), "metadata": metadatas[i]}
            for i, id_ in enumerate(ids)
        ]
        # Pinecone recommends batching upserts; fine at this project's scale
        # to send in chunks of 100.
        for start in range(0, len(items), 100):
            self._index.upsert(vectors=items[start:start + 100])

    def query(self, vector: np.ndarray, top_k: int = 5) -> List[Match]:
        result = self._index.query(
            vector=np.asarray(vector, dtype="float32").tolist(),
            top_k=top_k,
            include_metadata=True,
        )
        return [
            Match(id=m["id"], score=float(m["score"]), metadata=m.get("metadata", {}))
            for m in result.get("matches", [])
        ]

    def describe(self) -> dict:
        stats = self._index.describe_index_stats()
        return {
            "backend": "pinecone",
            "index_name": self.index_name,
            "vector_count": stats.get("total_vector_count", 0),
        }


def get_vector_store(dimension: int = 32) -> VectorStore:
    """
    Factory: returns a PineconeVectorStore if PINECONE_API_KEY is set,
    otherwise a LocalVectorStore. This is the single switch point between
    "real" cloud retrieval and local/offline development — nothing else in
    the codebase needs to know which backend it's talking to.
    """
    if os.environ.get("PINECONE_API_KEY"):
        return PineconeVectorStore(dimension=dimension)
    return LocalVectorStore()
