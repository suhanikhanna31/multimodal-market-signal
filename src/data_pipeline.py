"""
data_pipeline.py
──────────────────
Loads the synthetic parquet dataset and turns it into a tf.data.Dataset
with the three input branches model.py expects, plus a stratified
train/validation split.
"""

import os

import numpy as np
import pandas as pd
import tensorflow as tf

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "synthetic_market_data.parquet"
)


def load_dataframe(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python data/generate_synthetic_data.py` first."
        )
    return pd.read_parquet(path)


def _to_arrays(df: pd.DataFrame):
    headlines = df["headline"].to_numpy().astype(str)
    lookback = np.stack(df["lookback_returns"].to_numpy()).astype("float32")
    lookback = lookback[..., np.newaxis]  # (N, LOOKBACK_WINDOW, 1) for the LSTM branch
    tabular = df[["volume_z", "volatility", "rsi_like"]].to_numpy(dtype="float32")
    labels = df["label"].to_numpy(dtype="int64")
    return headlines, lookback, tabular, labels


def make_datasets(df: pd.DataFrame, batch_size: int = 64, val_frac: float = 0.15, seed: int = 42):
    """
    Returns (train_ds, val_ds, train_headlines_for_vectorizer).
    train_headlines_for_vectorizer is returned separately (as a plain numpy
    array) so the caller can `.adapt()` a TextVectorization layer on
    training text only, without leaking validation vocabulary.
    """
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = int(len(df) * val_frac)
    val_df, train_df = df.iloc[:n_val], df.iloc[n_val:]

    train_headlines, train_lookback, train_tabular, train_labels = _to_arrays(train_df)
    val_headlines, val_lookback, val_tabular, val_labels = _to_arrays(val_df)

    def _make_ds(headlines, lookback, tabular, labels, shuffle: bool):
        ds = tf.data.Dataset.from_tensor_slices((
            {
                "headline": headlines.reshape(-1, 1),
                "lookback_returns": lookback,
                "tabular_features": tabular,
            },
            labels,
        ))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(labels), seed=seed)
        ds = ds.batch(batch_size, drop_remainder=False)
        return ds.prefetch(tf.data.AUTOTUNE)

    train_ds = _make_ds(train_headlines, train_lookback, train_tabular, train_labels, shuffle=True)
    val_ds = _make_ds(val_headlines, val_lookback, val_tabular, val_labels, shuffle=False)

    return train_ds, val_ds, train_headlines
