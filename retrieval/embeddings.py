"""
retrieval/embeddings.py
────────────────────────
Turns headlines into semantic embedding vectors for retrieval.

Design choice: instead of pulling in a separate pretrained sentence-embedding
model (another dependency, another download), this reuses the 32-d
"text_embedding" layer that's already part of the multimodal fusion model
(src/model.py). If a trained model exists on disk, embeddings come from its
actual learned text encoder — the same weights driving the price-direction
predictions. If no trained model is found yet, this falls back to a freshly
built (untrained) embedder so the retrieval pipeline is still runnable
end-to-end; the vectors just won't carry a learned signal until you run
`python -m src.train_distributed` first. See README.md in this directory.
"""

import os

import numpy as np
import tensorflow as tf

from src.data_pipeline import load_dataframe
from src.model import build_model, build_text_embedder, build_text_vectorizer

DEFAULT_KERAS_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "saved_model.keras"
)

EMBEDDING_DIM = 32  # matches the "text_embedding" Dense layer in src/model.py


def load_text_embedder(keras_model_path: str = DEFAULT_KERAS_MODEL_PATH) -> tf.keras.Model:
    """
    Load the trained fusion model and slice out its text branch. Falls back
    to a freshly built (untrained) embedder if no trained model exists yet,
    printing a clear warning so callers don't silently treat random-init
    vectors as meaningful embeddings.
    """
    if os.path.exists(keras_model_path):
        full_model = tf.keras.models.load_model(keras_model_path)
        return build_text_embedder(full_model)

    print(
        f"[retrieval] No trained model found at {keras_model_path} — "
        "building an UNTRAINED text embedder instead. Run "
        "`python -m src.train_distributed` first for embeddings that "
        "actually carry a learned signal."
    )
    df = load_dataframe()
    text_vectorizer = build_text_vectorizer(df["headline"].to_numpy().astype(str))
    full_model = build_model(text_vectorizer)
    return build_text_embedder(full_model)


def embed_headlines(headlines, embedder: tf.keras.Model = None) -> np.ndarray:
    """
    headlines: iterable of strings.
    Returns an (N, EMBEDDING_DIM) float32 numpy array, L2-normalized so that
    cosine similarity (used by the vector store) reduces to a dot product.
    """
    if embedder is None:
        embedder = load_text_embedder()

    arr = np.array(list(headlines), dtype=str).reshape(-1, 1)
    # Keras 3's .predict() runs its input data-adapter path, which is fussy
    # about raw numpy string dtypes; a direct call (as src/model.py's own
    # tests do) sidesteps that and works whether the embedder came from a
    # loaded .keras model or a freshly built one.
    vectors = embedder(arr, training=False).numpy().astype("float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def embed_one(headline: str, embedder: tf.keras.Model = None) -> np.ndarray:
    """Convenience wrapper for embedding a single headline (used at serving time)."""
    return embed_headlines([headline], embedder=embedder)[0]
