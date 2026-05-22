"""Minimal production FastAPI for Air Paradis sentiment analysis."""

import json
import os
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.predictor import SentimentPredictor

DEBUG_LOG = Path(
    os.getenv(
        "DEBUG_LOG",
        "/Users/laureendademeule/Documents/Projets/P7/.cursor/debug-448d12.log",
    )
)
RUN_ID = os.getenv("RUN_ID", "azure-api")

predictor: SentimentPredictor | None = None
load_error: str | None = None
load_debug: dict | None = None
START_TIME = datetime.utcnow()


def _agent_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # #region agent log
    try:
        payload = {
            "sessionId": "448d12",
            "runId": RUN_ID,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(datetime.utcnow().timestamp() * 1000),
        }
        DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        with DEBUG_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
    except OSError:
        pass
    # #endregion


def _init_predictor() -> None:
    global predictor, load_error, load_debug
    root = Path(__file__).resolve().parents[2]
    production_dir = root / "models" / "production"
    marker = production_dir / "production_model.txt"
    bundle = production_dir / "tfidf_logistic_bundle.pkl"
    load_debug = {
        "root": str(root),
        "production_dir_exists": production_dir.exists(),
        "marker_exists": marker.exists(),
        "bundle_exists": bundle.exists(),
        "marker_value": marker.read_text().strip() if marker.exists() else None,
    }
    _agent_log("H1", "main.py:_init_predictor", "paths", load_debug)
    try:
        predictor = SentimentPredictor()
        if not predictor.bundle and predictor.bert_model is None and predictor.keras_model is None:
            raise RuntimeError("Model files found but bundle is empty after load")
        load_error = None
        _agent_log(
            "H2",
            "main.py:_init_predictor",
            "loaded",
            {"model_name": predictor.model_name, "bundle_keys": list(predictor.bundle.keys())},
        )
    except Exception as exc:
        predictor = None
        load_error = f"{type(exc).__name__}: {exc}"
        _agent_log("H3", "main.py:_init_predictor", "load_failed", {"error": load_error})


@asynccontextmanager
async def lifespan(app: FastAPI):
    if predictor is None and load_error is None:
        _init_predictor()
    yield


app = FastAPI(
    title="Air Paradis Sentiment API",
    description="Predict tweet sentiment for Air Paradis social monitoring",
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.getenv("SKIP_MODEL_INIT") != "1":
    _init_predictor()


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


@app.get("/health")
def health():
    return {
        "status": "healthy" if predictor else "degraded",
        "model_loaded": predictor is not None,
        "model": predictor.model_name if predictor else None,
        "load_error": load_error,
        "load_debug": load_debug,
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
