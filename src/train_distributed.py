"""
train_distributed.py
──────────────────────
Trains the multimodal fusion model using TensorFlow's native distributed
training API, `tf.distribute.MirroredStrategy` (synchronous data-parallel
training across all GPUs visible on one machine).

What this demonstrates
-------------------------
- Wrapping model construction + compilation in `strategy.scope()`, which is
  the part of the API that actually matters: it tells TF to create
  variables in a way that can be mirrored across replicas.
- Scaling batch size by `strategy.num_replicas_in_sync` (global batch size
  = per-replica batch size x number of replicas) — a detail that's easy to
  get wrong and skews results if missed.
- Configuring the `tf.data` pipeline's auto-sharding policy explicitly,
  which matters once you move to multi-worker training.

Honesty note on scope
------------------------
This runs on whatever hardware is available. Tested here on CPU only, so
`strategy.num_replicas_in_sync == 1` and this executes as a single
replica — but the code path is unchanged from what you'd run on a
multi-GPU box; only the visible device count changes.

For genuine multi-node training you'd switch to
`tf.distribute.MultiWorkerMirroredStrategy`, which needs a `TF_CONFIG`
environment variable on each worker describing the cluster, e.g.:

    TF_CONFIG='{
      "cluster": {"worker": ["host1:port", "host2:port"]},
      "task": {"type": "worker", "index": 0}
    }'

That's the TensorFlow-native equivalent of what `torchrun` +
FSDP/DeepSpeed config does in the PyTorch ecosystem — same goal
(coordinate gradient sync / parameter sharding across machines),
different framework, different config surface. It is *not* the same
mechanism as ZeRO-style parameter sharding (DeepSpeed/FSDP) or the
tensor/pipeline model-parallelism in Megatron-LM, which split a single
huge model across devices rather than replicating it — this script only
demonstrates the data-parallel case, which is what actually applies to a
model this size.
"""

import os

import tensorflow as tf

from src.data_pipeline import load_dataframe, make_datasets
from src.model import build_model, build_text_vectorizer

PER_REPLICA_BATCH_SIZE = 32
EPOCHS = 8
MODEL_OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_model")
KERAS_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "saved_model.keras")


def main():
    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of devices in sync: {strategy.num_replicas_in_sync}")

    global_batch_size = PER_REPLICA_BATCH_SIZE * strategy.num_replicas_in_sync

    df = load_dataframe()
    train_ds, val_ds, train_headlines = make_datasets(df, batch_size=global_batch_size)

    # Explicit sharding policy: DATA means each worker sees a shard of the
    # dataset rather than the whole thing. This is a no-op with one worker
    # but is the setting that matters once you move to
    # MultiWorkerMirroredStrategy.
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_ds = train_ds.with_options(options)
    val_ds = val_ds.with_options(options)

    # TextVectorization is adapted once, outside strategy.scope(), then
    # baked into the model — vocabulary/lookup tables aren't themselves
    # mirrored trainable variables.
    text_vectorizer = build_text_vectorizer(train_headlines)

    with strategy.scope():
        model = build_model(text_vectorizer)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
    )

    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    model.export(MODEL_OUT_DIR)
    print(f"Saved serving SavedModel to {MODEL_OUT_DIR}")

    # Also save the native Keras format (alongside the serving-only export
    # above). The serving SavedModel only exposes a prediction signature;
    # the .keras file keeps the full layer graph so retrieval/embeddings.py
    # can later slice out the trained "text_embedding" layer as its own
    # model (see src/model.py::build_text_embedder).
    model.save(KERAS_MODEL_PATH)
    print(f"Saved native Keras model to {KERAS_MODEL_PATH}")

    final_val_acc = history.history["val_accuracy"][-1]
    print(f"Final validation accuracy: {final_val_acc:.3f}")
    return model, history


if __name__ == "__main__":
    main()
