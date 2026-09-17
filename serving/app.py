"""
serving/app.py
────────────────────────────
A small FastAPI app for local deployment:

  GET  /health           - liveness check
  GET  /info              - which model/vector-store backend is loaded
  POST /predict           - next-day direction prediction for one example,
                             plus a retrieval-based contextual signal from
                             similar historical headlines

Run locally:
    uvicorn serving.app:app --reload --port 8000

Run in Docker:
    docker compose up --build

If no trained model exists yet (saved_model.keras missing), /predict
returns a 503 with instructions instead of crashing the app, so /health
and /info stay usable for container healthchecks.
"""

import os
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.model import LOOKBACK_WINDOW, build_text_embedder
from retrieval.vector_store import get_vector_store
from retrieval.query_similar import get_contextual_signal

KERAS_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "saved_model.keras")
LABEL_NAMES = {0: "down", 1: "flat", 2: "up"}

_state = {"model": None, "embedder": None, "store": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="Multimodal Market Signal API",
    description="Next-day price-direction prediction fusing headlines with price/volume features, "
                 "plus semantic retrieval of contextually similar historical headlines.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    headline: str = Field(..., json_schema_extra={"example": "SYN00 shares surges after quarterly results"})
    lookback_returns: List[float] = Field(
        ..., min_length=LOOKBACK_WINDOW, max_length=LOOKBACK_WINDOW,
        description=f"Last {LOOKBACK_WINDOW} daily returns, oldest first.",
    )
    volume_z: float
    volatility: float
    rsi_like: float
    top_k_similar: int = Field(5, ge=0, le=50, description="0 to skip retrieval and only run the model.")


class PredictResponse(BaseModel):
    direction: str
    probabilities: dict
    contextual_signal: Optional[dict] = None


def load_artifacts() -> None:
    if os.path.exists(KERAS_MODEL_PATH):
        _state["model"] = tf.keras.models.load_model(KERAS_MODEL_PATH)
        _state["embedder"] = build_text_embedder(_state["model"])
    else:
        print(
            f"[serving] No trained model found at {KERAS_MODEL_PATH}. "
            "/predict will return 503 until you run `python -m src.train_distributed`."
        )
    _state["store"] = get_vector_store()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/info")
def info() -> dict:
    return {
        "model_loaded": _state["model"] is not None,
        "model_path": KERAS_MODEL_PATH,
        "vector_store": _state["store"].describe() if _state["store"] else None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if _state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No trained model available. Run `python -m src.train_distributed` "
                "to produce saved_model.keras, then restart this service."
            ),
        )

    inputs = {
        "headline": np.array([[req.headline]], dtype=str),
        "lookback_returns": np.array(req.lookback_returns, dtype="float32").reshape(1, LOOKBACK_WINDOW, 1),
        "tabular_features": np.array([[req.volume_z, req.volatility, req.rsi_like]], dtype="float32"),
    }
    # See retrieval/embeddings.py for why this uses a direct call rather
    # than .predict() — Keras 3's predict() data-adapter path is fussy
    # about raw numpy string dtypes in a mixed input dict.
    probs = _state["model"](inputs, training=False).numpy()[0]
    direction = LABEL_NAMES[int(np.argmax(probs))]
    probabilities = {LABEL_NAMES[i]: round(float(p), 4) for i, p in enumerate(probs)}

    contextual_signal = None
    if req.top_k_similar > 0:
        contextual_signal = get_contextual_signal(
            req.headline,
            top_k=req.top_k_similar,
            embedder=_state["embedder"],
            store=_state["store"],
        )

    return PredictResponse(direction=direction, probabilities=probabilities, contextual_signal=contextual_signal)
