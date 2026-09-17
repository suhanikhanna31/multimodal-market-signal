"""
retrieval/build_index.py
────────────────────────────
Embeds every headline in the synthetic dataset and upserts it into the
vector store (Pinecone if PINECONE_API_KEY is set, otherwise the local
fallback — see vector_store.py), together with metadata (ticker, day,
label, the headline text itself) needed to turn a similarity match back
into a useful "contextual financial signal" at query time.

Run:
    python -m retrieval.build_index
    python -m retrieval.build_index --limit 500   # smaller/faster for local dev
"""

import argparse

from src.data_pipeline import load_dataframe
from retrieval.embeddings import embed_headlines, load_text_embedder
from retrieval.vector_store import get_vector_store

LABEL_NAMES = {0: "down", 1: "flat", 2: "up"}
BATCH_SIZE = 256


def build_index(limit: int = None, batch_size: int = BATCH_SIZE) -> dict:
    df = load_dataframe()
    if limit:
        df = df.iloc[:limit]

    embedder = load_text_embedder()
    store = get_vector_store()

    total = 0
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        vectors = embed_headlines(chunk["headline"].tolist(), embedder=embedder)
        ids = [f"{row.ticker}-{row.day}" for row in chunk.itertuples()]
        metadatas = [
            {
                "ticker": row.ticker,
                "day": int(row.day),
                "headline": row.headline,
                "label": int(row.label),
                "label_name": LABEL_NAMES[int(row.label)],
            }
            for row in chunk.itertuples()
        ]
        store.upsert(ids, vectors, metadatas)
        total += len(chunk)
        print(f"Upserted {total}/{len(df)} headlines...")

    info = store.describe()
    print(f"Done. Index info: {info}")
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Only index the first N rows (useful for a quick local demo).")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    build_index(limit=args.limit, batch_size=args.batch_size)
