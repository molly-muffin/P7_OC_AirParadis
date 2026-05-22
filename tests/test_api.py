"""Tests for production API."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api.main import app


@pytest.fixture
def client():
    mock_predictor = MagicMock()
    mock_predictor.model_name = "test_model"
    mock_predictor.predict.return_value = ("positive", 0.92)
    mock_predictor.info.return_value = {"model_name": "test_model"}

    with patch("api.main.predictor", mock_predictor):
        yield TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_endpoint(client):
    response = client.post("/predict", json={"text": "I love Air Paradis!"})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "positive"
    assert 0 <= data["confidence"] <= 1
    assert "prediction_id" in data


def test_predict_empty_text_rejected(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_feedback_endpoint(client):
    response = client.post(
        "/feedback",
        json={
            "text": "bad flight",
            "predicted_sentiment": "positive",
            "actual_sentiment": "negative",
        },
    )
    assert response.status_code == 200
    assert "status" in response.json()
