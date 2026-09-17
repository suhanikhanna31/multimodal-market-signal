"""
tests/test_model.py
──────────────────────
Shape and sanity tests for the multimodal fusion model. Not a full
training-quality test suite — the point is to catch structural breakage
(wrong input/output shapes, broken fusion) before it costs a slow
training run to notice.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import tensorflow as tf

from src.model import LOOKBACK_WINDOW, NUM_CLASSES, build_model, build_text_vectorizer


def _dummy_headlines():
    return np.array([
        "acme shares surge after quarterly results",
        "north stock slumps on analyst commentary",
        "blue holds steady ahead of earnings",
    ])


def test_text_vectorizer_adapts_and_has_vocab():
    vectorizer = build_text_vectorizer(_dummy_headlines(), max_tokens=50, sequence_length=8)
    assert vectorizer.vocabulary_size() > 2  # at least PAD/UNK + real tokens


def test_model_builds_and_has_expected_output_shape():
    vectorizer = build_text_vectorizer(_dummy_headlines(), max_tokens=50, sequence_length=8)
    model = build_model(vectorizer)

    batch_size = 4
    inputs = {
        "headline": np.array([["acme shares surge"]] * batch_size),
        "lookback_returns": np.random.randn(batch_size, LOOKBACK_WINDOW, 1).astype("float32"),
        "tabular_features": np.random.randn(batch_size, 3).astype("float32"),
    }
    output = model(inputs, training=False)
    assert output.shape == (batch_size, NUM_CLASSES)


def test_model_output_is_a_valid_probability_distribution():
    vectorizer = build_text_vectorizer(_dummy_headlines(), max_tokens=50, sequence_length=8)
    model = build_model(vectorizer)

    inputs = {
        "headline": np.array([["blue trades flat"]]),
        "lookback_returns": np.random.randn(1, LOOKBACK_WINDOW, 1).astype("float32"),
        "tabular_features": np.random.randn(1, 3).astype("float32"),
    }
    output = model(inputs, training=False).numpy()[0]
    assert np.isclose(output.sum(), 1.0, atol=1e-4)
    assert (output >= 0).all()


def test_model_is_trainable_for_one_step():
    vectorizer = build_text_vectorizer(_dummy_headlines(), max_tokens=50, sequence_length=8)
    model = build_model(vectorizer)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

    batch_size = 4
    inputs = {
        "headline": np.array([["acme shares surge"]] * batch_size),
        "lookback_returns": np.random.randn(batch_size, LOOKBACK_WINDOW, 1).astype("float32"),
        "tabular_features": np.random.randn(batch_size, 3).astype("float32"),
    }
    labels = np.random.randint(0, NUM_CLASSES, size=(batch_size,))

    # Keras 3's fit() doesn't accept raw numpy string arrays directly in a
    # dict of inputs; route through tf.data.Dataset as the real training
    # pipeline (src/data_pipeline.py) does.
    ds = tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(batch_size)
    history = model.fit(ds, epochs=1, verbose=0)
    assert "loss" in history.history
    assert np.isfinite(history.history["loss"][0])
