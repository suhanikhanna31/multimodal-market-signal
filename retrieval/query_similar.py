"""
retrieval/query_similar.py
────────────────────────────
Given a new headline, embed it and retrieve the most semantically similar
historical headlines from the vector store, then summarize their outcomes
into a "contextual financial signal": e.g. "8 of the 10 most similar past
headlines for this kind of language preceded an up day". This is the piece
that connects the retrieval workflow back to something a caller (or
serving/app.py) can actually use alongside the model's own prediction.

This does not feed the retrieved context back into the trained model as a
new input tensor — the model's inputs are fixed (headline, lookback
returns, tabular features), so the model is not "retrieval-augmented" in
the RAG-for-generation sense. What's connected here is model -> embedding
-> retrieval -> a contextual summary shown alongside the prediction. See
README.md for the honesty note.
"""

from collections import Counter
from typing import Optional

import numpy as np

from retrieval.embeddings import embed_one, load_text_embedder
from retrieval.vector_store import VectorStore, get_vector_store

LABEL_NAMES = {0: "down", 1: "flat", 2: "up"}


def get_contextual_signal(
    headline: str,
    top_k: int = 10,
    embedder=None,
    store: Optional[VectorStore] = None,
) -> dict:
    """
    Returns:
        {
          "query_headline": str,
          "matches": [{"headline": ..., "score": ..., "label_name": ..., "ticker": ..., "day": ...}, ...],
          "label_distribution": {"down": n, "flat": n, "up": n},
          "majority_label": "up" | "down" | "flat" | None,
        }
    """
    embedder = embedder or load_text_embedder()
    store = store or get_vector_store()

    vector = embed_one(headline, embedder=embedder)
    matches = store.query(vector, top_k=top_k)

    label_counts = Counter(m.metadata.get("label_name") for m in matches if m.metadata)
    majority_label = label_counts.most_common(1)[0][0] if label_counts else None

    return {
        "query_headline": headline,
        "matches": [
            {
                "headline": m.metadata.get("headline"),
                "score": round(m.score, 4),
                "label_name": m.metadata.get("label_name"),
                "ticker": m.metadata.get("ticker"),
                "day": m.metadata.get("day"),
            }
            for m in matches
        ],
        "label_distribution": dict(label_counts),
        "majority_label": majority_label,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("headline", type=str, help="Headline to find contextual matches for.")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    signal = get_contextual_signal(args.headline, top_k=args.top_k)
    print(f"Query: {signal['query_headline']}\n")
    for m in signal["matches"]:
        print(f"  [{m['score']:.3f}] ({m['label_name']:>4}) {m['headline']}")
    print(f"\nLabel distribution among top matches: {signal['label_distribution']}")
    print(f"Majority label: {signal['majority_label']}")
