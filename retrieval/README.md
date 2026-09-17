# Semantic retrieval for contextual financial signals

Turns headlines into embeddings and stores them in a vector database
(Pinecone, with a local fallback), so a new headline can be matched
against semantically similar historical headlines and their outcomes.

## Pipeline

```
headline (string)
   │
   ▼
retrieval/embeddings.py      — 32-d embedding via the trained model's own
                                text branch (src/model.py::build_text_embedder)
   │
   ▼
retrieval/vector_store.py    — upsert / cosine-similarity query
                                (PineconeVectorStore or LocalVectorStore)
   │
   ▼
retrieval/query_similar.py   — turns the top-k matches into a contextual
                                signal: label distribution + majority label
                                among historically similar headlines
```

## Why the embedder reuses the trained model instead of a separate model

Rather than adding a second pretrained sentence-embedding model (another
dependency, another download, another vocabulary disconnected from this
project's data), `retrieval/embeddings.py` slices the 32-d
`text_embedding` layer straight out of the trained fusion model. That
means:

- The embedding space is literally the same representation the model
  learned to predict price direction from, so "similar headlines" means
  "similar to this model," not "similar according to some unrelated
  general-purpose encoder."
- No extra model weights to download or version.
- The trade-off: if you haven't run `python -m src.train_distributed`
  yet, `load_text_embedder()` falls back to a freshly initialized
  (untrained) embedder so the rest of the pipeline is still runnable —
  but those vectors are random and **not semantically meaningful**. The
  fallback prints an explicit warning when this happens; don't mistake
  its output for real retrieval quality.

## Why there's a local fallback vector store, not just Pinecone

`retrieval/vector_store.py::get_vector_store()` picks a backend purely
from environment variables:

- `PINECONE_API_KEY` set → `PineconeVectorStore` (real Pinecone index,
  created on first use).
- Not set → `LocalVectorStore` (numpy cosine-similarity search, persisted
  to `retrieval/local_index/*.npy` + `*.json`).

This is what makes `docker compose up` and the test suite work without a
Pinecone account, API key, or network access — the exact same code path
(`build_index.py`, `query_similar.py`, `serving/app.py`) runs against
either backend. The local store is a brute-force scan, fine at this
project's scale (a few thousand headlines) and not intended to replace a
real ANN index at production scale.

## What this retrieval workflow does *not* do

It does not feed retrieved context back into the trained model as a new
input tensor — the model's three inputs (headline, lookback returns,
tabular features) are fixed at training time. What's built here is
model → embedding → retrieval → a contextual summary shown *alongside*
the model's prediction (see `serving/app.py`'s `/predict` response), not
a retrieval-augmented generation loop that changes what the model itself
sees.

## Running it

```bash
# 1. Train the model first (produces saved_model.keras, used for embeddings)
python -m src.train_distributed

# 2. Embed the dataset and build the index (local backend by default)
python -m retrieval.build_index --limit 2000

# 3. Query it
python -m retrieval.query_similar "SYN00 shares surges after quarterly results"
```

To use real Pinecone instead of the local fallback, set `PINECONE_API_KEY`
(and optionally `PINECONE_INDEX_NAME`, `PINECONE_CLOUD`, `PINECONE_REGION`)
before running the same commands — see `.env.example`.
