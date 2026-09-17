"""
quantize_and_benchmark.py
────────────────────────────
Inference optimization: converts the trained SavedModel to TFLite with
post-training dynamic-range quantization, then benchmarks the quantized
model against the original on the same batch of inputs — real numbers,
not estimates:

  - On-disk size (SavedModel dir vs .tflite file)
  - Per-sample inference latency (Keras/TF runtime vs TFLite interpreter)

Run `python -m src.train_distributed` first to produce ../saved_model.
"""

import os
import time

import numpy as np
import tensorflow as tf

from src.data_pipeline import load_dataframe, make_datasets

SAVED_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_model")
TFLITE_PATH = os.path.join(os.path.dirname(__file__), "..", "model_quantized.tflite")
BENCHMARK_N = 200


def _dir_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def convert_to_tflite():
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
    # Dynamic-range quantization: weights stored as int8, activations
    # computed in float at inference time. Doesn't need a representative
    # dataset (unlike full-integer quantization), and is the safest
    # default speed/size win for a model this small.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # The Bidirectional LSTM branch lowers to a dynamic TensorList op that
    # TFLite's built-in op set can't legalize on its own. Falling back to
    # TF's op kernels (SELECT_TF_OPS) for just that op is the documented
    # fix — everything else in the graph still converts to native TFLite
    # ops, so this only costs a slightly larger runtime, not the whole
    # size/speed benefit of quantization.
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()
    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)
    return TFLITE_PATH


# Note on what SELECT_TF_OPS actually costs: the text branch's TextVectorization
# and the Bidirectional LSTM lower to a handful of ops (string splitting,
# TensorList reserve/stack, ragged-to-dense) that aren't in TFLite's builtin
# op set and get delegated to the "Flex" TF runtime instead of native TFLite
# kernels. That's why this still needs the full TensorFlow package at
# inference time (or the tflite-runtime + flex-delegate build) rather than
# the minimal tflite-runtime-only deployment you'd get with a pure-numeric
# model. Worth knowing before claiming this is "edge-deployable" as-is —
# it's real quantization and a real, measured speed/size win, but the text
# branch keeps it dependent on TF's runtime rather than the lightest-weight
# TFLite deployment path.


def _benchmark_keras(saved_model_dir: str, inputs: dict, n_runs: int) -> float:
    infer = tf.saved_model.load(saved_model_dir).signatures["serving_default"]
    # Warm-up (first call pays graph-tracing / lazy-init cost)
    infer(**inputs)
    start = time.perf_counter()
    for _ in range(n_runs):
        infer(**inputs)
    elapsed = time.perf_counter() - start
    return elapsed / n_runs


def _benchmark_tflite(tflite_path: str, inputs: dict, n_runs: int) -> float:
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = {d["name"]: d for d in interpreter.get_input_details()}

    def _set_inputs():
        for name, detail in input_details.items():
            # TFLite input tensor names are like "serving_default_headline:0"
            key = next(k for k in inputs if k in name)
            interpreter.set_tensor(detail["index"], inputs[key])

    _set_inputs()
    interpreter.invoke()  # warm-up

    start = time.perf_counter()
    for _ in range(n_runs):
        _set_inputs()
        interpreter.invoke()
    elapsed = time.perf_counter() - start
    return elapsed / n_runs


def main():
    if not os.path.exists(SAVED_MODEL_DIR):
        raise FileNotFoundError(
            f"{SAVED_MODEL_DIR} not found. Run `python -m src.train_distributed` first."
        )

    tflite_path = convert_to_tflite()

    df = load_dataframe()
    _, val_ds, _ = make_datasets(df, batch_size=1)
    single_batch = next(iter(val_ds.take(1)))[0]
    single_batch = {k: v.numpy() for k, v in single_batch.items()}

    keras_size = _dir_size_bytes(SAVED_MODEL_DIR)
    tflite_size = os.path.getsize(tflite_path)

    keras_latency = _benchmark_keras(SAVED_MODEL_DIR, single_batch, BENCHMARK_N)
    tflite_latency = _benchmark_tflite(tflite_path, single_batch, BENCHMARK_N)

    print("── Size ─────────────────────────────────────────────")
    print(f"SavedModel dir : {keras_size / 1024:.1f} KB")
    print(f"TFLite file    : {tflite_size / 1024:.1f} KB")
    print(f"Size reduction : {(1 - tflite_size / keras_size) * 100:.1f}%")
    print()
    print(f"── Latency (avg over {BENCHMARK_N} single-sample calls) ──")
    print(f"Keras/TF runtime : {keras_latency * 1000:.3f} ms/sample")
    print(f"TFLite runtime   : {tflite_latency * 1000:.3f} ms/sample")
    print(f"Speedup          : {keras_latency / tflite_latency:.2f}x")


if __name__ == "__main__":
    main()
