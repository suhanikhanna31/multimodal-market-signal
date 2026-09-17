#!/bin/sh
# start.sh
# ────────────────────────
# Hugging Face Spaces (and any host without a persisted volume) gives this
# container a fresh filesystem on every restart, so saved_model.keras and
# the retrieval index won't survive a redeploy unless they're baked into
# the image or regenerated on boot. This does the latter: train + index
# once if the artifacts aren't already present, then start the API.
#
# This makes cold starts slower (roughly the time src/train_distributed.py
# takes on CPU, well under a minute on this project's small synthetic
# dataset) but keeps the image itself simple and rebuild-free.

set -e

if [ ! -f "data/synthetic_market_data.parquet" ]; then
  echo "[start.sh] Generating synthetic dataset..."
  python data/generate_synthetic_data.py
fi

if [ ! -f "saved_model.keras" ]; then
  echo "[start.sh] No trained model found — training now (one-time, on boot)..."
  python -m src.train_distributed
fi

if [ ! -d "retrieval/local_index" ] || [ -z "$(ls -A retrieval/local_index 2>/dev/null)" ]; then
  echo "[start.sh] No retrieval index found — building now (one-time, on boot)..."
  python -m retrieval.build_index --limit 2000
fi

echo "[start.sh] Starting API on port ${PORT:-7860}..."
exec uvicorn serving.app:app --host 0.0.0.0 --port "${PORT:-7860}"
