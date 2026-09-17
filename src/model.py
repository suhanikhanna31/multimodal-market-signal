"""
model.py
─────────
A multimodal fusion model with three input branches:

  1. Text branch      — headline -> TextVectorization -> Embedding ->
                         Bidirectional LSTM -> 32-d vector.
  2. Sequence branch   — the last LOOKBACK_WINDOW daily returns ->
                         LSTM -> 16-d vector (captures short-term momentum
                         shape, not just a single summary stat).
  3. Tabular branch    — same-day volume_z / volatility / rsi_like ->
                         Dense -> 16-d vector.

The three vectors are concatenated and passed through a small classifier
head predicting next-day direction: {down, flat, up}.

Kept intentionally small (a few hundred thousand parameters, not millions)
so it trains in seconds on CPU — the point of this project is the
end-to-end pipeline (multimodal fusion -> distributed training ->
quantization), not squeezing out benchmark accuracy.
"""

import tensorflow as tf
from tensorflow.keras import layers

LOOKBACK_WINDOW = 10
NUM_CLASSES = 3


def build_text_vectorizer(headlines, max_tokens: int = 2000, sequence_length: int = 16):
    """Adapt a TextVectorization layer on the training headlines."""
    vectorizer = layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=sequence_length,
    )
    vectorizer.adapt(headlines)
    return vectorizer


def build_model(text_vectorizer: layers.TextVectorization) -> tf.keras.Model:
    vocab_size = text_vectorizer.vocabulary_size()

    # ── Text branch ───────────────────────────────────────────────────────
    text_input = tf.keras.Input(shape=(1,), dtype=tf.string, name="headline")
    x_text = text_vectorizer(text_input)
    x_text = layers.Embedding(vocab_size, 32, mask_zero=True)(x_text)
    x_text = layers.Bidirectional(layers.LSTM(16))(x_text)
    x_text = layers.Dense(32, activation="relu", name="text_embedding")(x_text)

    # ── Return-sequence branch ───────────────────────────────────────────
    seq_input = tf.keras.Input(shape=(LOOKBACK_WINDOW, 1), name="lookback_returns")
    x_seq = layers.LSTM(16)(seq_input)
    x_seq = layers.Dense(16, activation="relu", name="sequence_embedding")(x_seq)

    # ── Tabular same-day features branch ─────────────────────────────────
    tab_input = tf.keras.Input(shape=(3,), name="tabular_features")
    x_tab = layers.Dense(16, activation="relu")(tab_input)
    x_tab = layers.Dense(16, activation="relu", name="tabular_embedding")(x_tab)

    # ── Fusion + classifier head ─────────────────────────────────────────
    fused = layers.Concatenate(name="fusion")([x_text, x_seq, x_tab])
    x = layers.Dense(32, activation="relu")(fused)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(NUM_CLASSES, activation="softmax", name="direction")(x)

    model = tf.keras.Model(
        inputs={
            "headline": text_input,
            "lookback_returns": seq_input,
            "tabular_features": tab_input,
        },
        outputs=output,
        name="multimodal_market_signal",
    )
    return model


def build_text_embedder(full_model: tf.keras.Model) -> tf.keras.Model:
    """
    Slice the text branch out of a built/trained `full_model` and return it
    as its own Model: headline (string) -> 32-d "text_embedding" vector.

    This is the model used for semantic retrieval (see retrieval/embeddings.py):
    rather than pulling in a separate pretrained sentence-embedding model, we
    reuse the same text encoder the fusion model was trained with, so the
    embedding space is directly tied to the signal the model itself learned
    from headlines. If `full_model` hasn't been trained yet, this still
    returns a structurally valid embedder — its vectors just won't carry a
    learned signal yet (see retrieval/README.md for the honesty note).
    """
    return tf.keras.Model(
        inputs=full_model.input["headline"],
        outputs=full_model.get_layer("text_embedding").output,
        name="text_embedder",
    )
