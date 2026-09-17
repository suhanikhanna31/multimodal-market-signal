"""
tests/test_serving.py
──────────────────────────
Exercises the FastAPI app's health/info endpoints, and confirms /predict
degrades to a clear 503 (rather than crashing) when no trained model is
present — which is the state of a fresh checkout / CI runner that hasn't
run training.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient

from serving.app import app
from src.model import LOOKBACK_WINDOW


@pytest.fixture(scope="module")
def client():
    # Using TestClient as a context manager triggers FastAPI's startup
    # event (load_artifacts), same as a real app boot.
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_info_reports_backend(client):
    response = client.get("/info")
    assert response.status_code == 200
    body = response.json()
    assert "model_loaded" in body
    assert "vector_store" in body
    assert body["vector_store"]["backend"] in {"local", "pinecone"}


def test_predict_without_trained_model_returns_503(client):
    # Fresh checkouts / CI have no saved_model.keras yet.
    payload = {
        "headline": "SYN00 shares surges after quarterly results",
        "lookback_returns": [0.001] * LOOKBACK_WINDOW,
        "volume_z": 0.1,
        "volatility": 0.02,
        "rsi_like": 0.5,
        "top_k_similar": 3,
    }
    response = client.post("/predict", json=payload)
    if response.status_code == 200:
        # If a previous test/session trained a model on disk, that's a
        # valid state too — just check the response shape.
        body = response.json()
        assert body["direction"] in {"down", "flat", "up"}
        assert set(body["probabilities"].keys()) == {"down", "flat", "up"}
    else:
        assert response.status_code == 503
        assert "train_distributed" in response.json()["detail"]


def test_predict_rejects_wrong_length_lookback(client):
    payload = {
        "headline": "SYN00 shares surges after quarterly results",
        "lookback_returns": [0.001, 0.002],  # wrong length
        "volume_z": 0.1,
        "volatility": 0.02,
        "rsi_like": 0.5,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
