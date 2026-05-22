"""Minimal production FastAPI for Air Paradis sentiment analysis."""

import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.predictor import SentimentPredictor

app = FastAPI(
    title="Air Paradis Sentiment API",
    description="Predict tweet sentiment for Air Paradis social monitoring",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor: SentimentPredictor | None = None
START_TIME = datetime.utcnow()


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)


class PredictResponse(BaseModel):
    sentiment: str
    confidence: float
    model: str
    prediction_id: str


class FeedbackRequest(BaseModel):
    text: str
    predicted_sentiment: str
    actual_sentiment: str
    prediction_id: str | None = None


@app.on_event("startup")
def load_model():
    global predictor
    try:
        predictor = SentimentPredictor()
    except Exception as exc:
        predictor = None
        print(f"Warning: could not load model: {exc}")


@app.get("/health")
def health():
    return {
        "status": "healthy" if predictor else "degraded",
        "model_loaded": predictor is not None,
        "model": predictor.model_name if predictor else None,
        "uptime_seconds": (datetime.utcnow() - START_TIME).total_seconds(),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    label, confidence = predictor.predict(request.text)
    return PredictResponse(
        sentiment=label,
        confidence=round(confidence, 4),
        model=predictor.model_name,
        prediction_id=str(uuid.uuid4()),
    )


@app.get("/model/info")
def model_info():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return predictor.info()


@app.post("/feedback")
def feedback(request: FeedbackRequest):
    if request.predicted_sentiment.lower() != request.actual_sentiment.lower():
        try:
            from monitoring.app_insights import check_alert_threshold, track_misclassification

            track_misclassification(
                tweet_text=request.text,
                predicted_sentiment=request.predicted_sentiment,
                user_feedback=request.actual_sentiment,
            )
            alert = check_alert_threshold()
            return {"status": "logged", "alert_triggered": alert}
        except ImportError:
            return {"status": "logged_local", "alert_triggered": False}
    return {"status": "ok", "alert_triggered": False}
