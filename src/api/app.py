"""
FastAPI application for Air Paradis sentiment analysis.

This application provides:
- Sentiment prediction endpoints
- Health monitoring
- Feedback collection
- Model management
- Performance metrics
"""

import os
import sys
import time
import uuid
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from api.schemas import (
    PredictionRequest, PredictionResponse, BatchPredictionRequest, BatchPredictionResponse,
    FeedbackRequest, FeedbackResponse, HealthResponse, ModelInfoResponse,
    ErrorResponse, MetricsResponse, ConfigRequest, ConfigResponse,
    SentimentLabel
)
from data.preprocessor import TextPreprocessor
from models.logistic_model import LogisticSentimentModel
from models.deep_learning_model import DeepLearningModel
from models.bert_model import BertSentimentModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
models: Dict[str, Any] = {}
preprocessor: TextPreprocessor = None
app_start_time: datetime = datetime.utcnow()
prediction_cache: Dict[str, Any] = {}
feedback_storage: List[Dict[str, Any]] = []
metrics_storage: Dict[str, Any] = {
    "total_predictions": 0,
    "predictions_by_hour": {},
    "response_times": [],
    "errors": 0,
    "model_usage": {}
}

# Configuration
config = {
    "default_model": "best",
    "confidence_threshold": 0.5,
    "max_text_length": 1000,
    "enable_logging": True,
    "cache_predictions": True,
    "cache_ttl_minutes": 60
}


async def load_models():
    """Load all available models."""
    global models, preprocessor
    
    logger.info("Loading models and preprocessor...")
    
    try:
        # Initialize preprocessor
        preprocessor = TextPreprocessor()
        
        # Try to load vectorizers
        try:
            preprocessor.load_vectorizers()
            logger.info("Loaded cached vectorizers")
        except:
            logger.warning("No cached vectorizers found")
        
        # Model directory
        model_dir = Path("models")
        
        # Load Logistic Regression model
        try:
            logistic_model = LogisticSentimentModel(model_dir=str(model_dir))
            logistic_path = model_dir / "logistic_regression_model.pkl"
            if logistic_path.exists():
                logistic_model.load_model(str(logistic_path))
                models["logistic"] = logistic_model
                logger.info("Loaded Logistic Regression model")
        except Exception as e:
            logger.warning(f"Could not load Logistic Regression model: {str(e)}")
        
        # Load Deep Learning models
        for model_type in ["lstm", "cnn", "hybrid"]:
            try:
                dl_model = DeepLearningModel(model_dir=str(model_dir))
                model_path = model_dir / f"{model_type}_model.h5"
                if model_path.exists():
                    dl_model.load_model(str(model_path))
                    models[model_type] = dl_model
                    logger.info(f"Loaded {model_type.upper()} model")
            except Exception as e:
                logger.warning(f"Could not load {model_type} model: {str(e)}")
        
        # Load BERT models
        for model_name in ["bert_base_uncased", "distilbert_base_uncased"]:
            try:
                bert_model = BertSentimentModel(model_dir=str(model_dir))
                model_path = model_dir / f"{model_name}_finetuned"
                if model_path.exists():
                    bert_model.load_model(str(model_path))
                    models[model_name] = bert_model
                    logger.info(f"Loaded {model_name} model")
            except Exception as e:
                logger.warning(f"Could not load {model_name} model: {str(e)}")
        
        # Set best model
        if models:
            # Simple heuristic: prefer BERT > Deep Learning > Logistic
            if "distilbert_base_uncased" in models:
                models["best"] = models["distilbert_base_uncased"]
            elif "bert_base_uncased" in models:
                models["best"] = models["bert_base_uncased"]
            elif "hybrid" in models:
                models["best"] = models["hybrid"]
            elif "lstm" in models:
                models["best"] = models["lstm"]
            elif "logistic" in models:
                models["best"] = models["logistic"]
            
            logger.info(f"Set best model: {config['default_model']}")
        
        logger.info(f"Loaded {len(models)} models successfully")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        # Continue with empty models dict - API will return appropriate errors


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    await load_models()
    yield
    # Shutdown
    logger.info("Shutting down application...")


# Create FastAPI app
app = FastAPI(
    title="Air Paradis Sentiment Analysis API",
    description="Production-ready sentiment analysis API for social media monitoring",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)


# Utility functions
def get_model(model_type: str = None) -> Any:
    """Get model by type."""
    if not models:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No models are currently loaded"
        )
    
    model_key = model_type or config["default_model"]
    
    if model_key not in models:
        available_models = list(models.keys())
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{model_key}' not available. Available models: {available_models}"
        )
    
    return models[model_key]


def update_metrics(model_used: str, response_time: float, error: bool = False):
    """Update API metrics."""
    current_hour = datetime.utcnow().strftime("%Y-%m-%d-%H")
    
    metrics_storage["total_predictions"] += 1
    
    if current_hour not in metrics_storage["predictions_by_hour"]:
        metrics_storage["predictions_by_hour"][current_hour] = 0
    metrics_storage["predictions_by_hour"][current_hour] += 1
    
    metrics_storage["response_times"].append(response_time)
    # Keep only last 1000 response times
    if len(metrics_storage["response_times"]) > 1000:
        metrics_storage["response_times"] = metrics_storage["response_times"][-1000:]
    
    if error:
        metrics_storage["errors"] += 1
    
    if model_used not in metrics_storage["model_usage"]:
        metrics_storage["model_usage"][model_used] = 0
    metrics_storage["model_usage"][model_used] += 1


def cache_key(text: str, model_type: str) -> str:
    """Generate cache key for prediction."""
    import hashlib
    return hashlib.md5(f"{text}:{model_type}".encode()).hexdigest()


def get_cached_prediction(text: str, model_type: str) -> Optional[Dict[str, Any]]:
    """Get cached prediction if available and not expired."""
    if not config["cache_predictions"]:
        return None
    
    key = cache_key(text, model_type)
    if key in prediction_cache:
        cached = prediction_cache[key]
        if datetime.utcnow() - cached["timestamp"] < timedelta(minutes=config["cache_ttl_minutes"]):
            return cached["prediction"]
        else:
            # Remove expired cache entry
            del prediction_cache[key]
    
    return None


def cache_prediction(text: str, model_type: str, prediction: Dict[str, Any]):
    """Cache prediction result."""
    if not config["cache_predictions"]:
        return
    
    key = cache_key(text, model_type)
    prediction_cache[key] = {
        "prediction": prediction,
        "timestamp": datetime.utcnow()
    }
    
    # Clean old cache entries (keep last 1000)
    if len(prediction_cache) > 1000:
        oldest_keys = sorted(prediction_cache.keys(), 
                           key=lambda k: prediction_cache[k]["timestamp"])[:100]
        for old_key in oldest_keys:
            del prediction_cache[old_key]


async def predict_sentiment(text: str, model_type: str = None) -> Dict[str, Any]:
    """Core sentiment prediction function."""
    start_time = time.time()
    
    try:
        # Check cache first
        cached_result = get_cached_prediction(text, model_type or config["default_model"])
        if cached_result:
            return cached_result
        
        # Get model
        model = get_model(model_type)
        model_name = model_type or config["default_model"]
        
        # Preprocess text
        cleaned_text = preprocessor.clean_text(text)
        
        # Make prediction based on model type
        if hasattr(model, 'predict_proba'):  # Sklearn model
            # Create DataFrame for sklearn models
            import pandas as pd
            processed_df = preprocessor.preprocess_dataframe(
                pd.DataFrame({'text': [text], 'target': [0]})
            )
            
            prediction = model.predict(processed_df)[0]
            probabilities = model.predict_proba(processed_df)[0]
            
            sentiment = SentimentLabel.POSITIVE if prediction == 1 else SentimentLabel.NEGATIVE
            confidence = float(max(probabilities))
            prob_dict = {
                "negative": float(probabilities[0]),
                "positive": float(probabilities[1])
            }
            
        elif hasattr(model, 'predict') and hasattr(model, 'tokenizer'):  # Deep learning model
            # Prepare sequence
            X, _ = model.prepare_data([cleaned_text], fit_tokenizer=False)
            
            prediction_proba = model.predict(X)[0]
            prediction = int(prediction_proba > 0.5)
            
            sentiment = SentimentLabel.POSITIVE if prediction == 1 else SentimentLabel.NEGATIVE
            confidence = float(prediction_proba if prediction == 1 else 1 - prediction_proba)
            prob_dict = {
                "negative": float(1 - prediction_proba),
                "positive": float(prediction_proba)
            }
            
        elif hasattr(model, 'predict') and hasattr(model, 'model_name'):  # BERT model
            predictions, probabilities = model.predict([cleaned_text])
            
            prediction = predictions[0]
            sentiment = SentimentLabel.POSITIVE if prediction == 1 else SentimentLabel.NEGATIVE
            confidence = float(max(probabilities[0]))
            prob_dict = {
                "negative": float(probabilities[0][0]),
                "positive": float(probabilities[0][1])
            }
            
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model does not support prediction"
            )
        
        processing_time = (time.time() - start_time) * 1000
        
        result = {
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence,
            "probabilities": prob_dict,
            "model_used": model_name,
            "processing_time_ms": processing_time,
            "timestamp": datetime.utcnow()
        }
        
        # Cache result
        cache_prediction(text, model_name, result)
        
        # Update metrics
        update_metrics(model_name, processing_time)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        update_metrics(model_type or config["default_model"], processing_time, error=True)
        
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Air Paradis Sentiment Analysis API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = (datetime.utcnow() - app_start_time).total_seconds()
    
    return HealthResponse(
        status="healthy" if models else "degraded",
        version="1.0.0",
        models_loaded=list(models.keys()),
        uptime_seconds=uptime
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict sentiment for a single text."""
    if len(request.text) > config["max_text_length"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Text too long. Maximum length: {config['max_text_length']}"
        )
    
    result = await predict_sentiment(request.text, request.model_type)
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict sentiment for multiple texts."""
    start_time = time.time()
    
    # Validate total text length
    total_length = sum(len(text) for text in request.texts)
    if total_length > config["max_text_length"] * len(request.texts):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Total text length exceeds limits"
        )
    
    # Process predictions
    predictions = []
    for text in request.texts:
        try:
            result = await predict_sentiment(text, request.model_type)
            predictions.append(PredictionResponse(**result))
        except Exception as e:
            logger.error(f"Batch prediction error for text '{text[:50]}...': {str(e)}")
            # Continue with other predictions
    
    total_time = (time.time() - start_time) * 1000
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_count=len(predictions),
        model_used=request.model_type or config["default_model"],
        total_processing_time_ms=total_time
    )


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest, background_tasks: BackgroundTasks):
    """Submit feedback on predictions."""
    feedback_id = str(uuid.uuid4())
    
    feedback_data = {
        "id": feedback_id,
        "text": request.text,
        "predicted_sentiment": request.predicted_sentiment,
        "actual_sentiment": request.actual_sentiment,
        "confidence": request.confidence,
        "model_used": request.model_used,
        "user_id": request.user_id,
        "comments": request.comments,
        "timestamp": datetime.utcnow()
    }
    
    # Store feedback (in production, use a database)
    feedback_storage.append(feedback_data)
    
    # Process feedback in background (e.g., retrain model, update metrics)
    background_tasks.add_task(process_feedback, feedback_data)
    
    return FeedbackResponse(
        message="Feedback received successfully",
        feedback_id=feedback_id
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about loaded models."""
    if not models:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No models are currently loaded"
        )
    
    # Get current model info
    current_model_key = config["default_model"]
    current_model = models.get(current_model_key)
    
    if current_model:
        current_info = {
            "name": current_model_key,
            "type": getattr(current_model, 'model_type', 'unknown'),
            "version": "1.0.0",
            "status": "loaded"
        }
        
        # Add model-specific info
        if hasattr(current_model, 'get_model_info'):
            model_info = current_model.get_model_info()
            if 'training_history' in model_info:
                history = model_info['training_history']
                if 'val_metrics' in history:
                    current_info["accuracy"] = history['val_metrics'].get('accuracy')
                    current_info["f1_score"] = history['val_metrics'].get('f1')
    else:
        current_info = {
            "name": current_model_key,
            "type": "unknown",
            "version": "1.0.0",
            "status": "not_loaded"
        }
    
    # Get all available models info
    available_models = []
    for model_key, model in models.items():
        if model_key == "best":
            continue
            
        model_info = {
            "name": model_key,
            "type": getattr(model, 'model_type', 'unknown'),
            "version": "1.0.0",
            "status": "loaded"
        }
        available_models.append(model_info)
    
    return ModelInfoResponse(
        current_model=current_info,
        available_models=available_models
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get API performance metrics."""
    current_time = datetime.utcnow()
    one_hour_ago = current_time - timedelta(hours=1)
    one_day_ago = current_time - timedelta(days=1)
    
    # Calculate predictions in last hour and day
    predictions_last_hour = 0
    predictions_last_day = 0
    
    for hour_str, count in metrics_storage["predictions_by_hour"].items():
        try:
            hour_time = datetime.strptime(hour_str, "%Y-%m-%d-%H")
            if hour_time >= one_hour_ago:
                predictions_last_hour += count
            if hour_time >= one_day_ago:
                predictions_last_day += count
        except:
            continue
    
    # Calculate average response time
    avg_response_time = 0
    if metrics_storage["response_times"]:
        avg_response_time = sum(metrics_storage["response_times"]) / len(metrics_storage["response_times"])
    
    # Calculate error rate
    total_requests = metrics_storage["total_predictions"]
    error_rate = 0
    if total_requests > 0:
        error_rate = metrics_storage["errors"] / total_requests
    
    # Calculate uptime
    uptime_seconds = (current_time - app_start_time).total_seconds()
    uptime_percentage = 100.0  # Simplified - in production, track actual downtime
    
    return MetricsResponse(
        total_predictions=total_requests,
        predictions_last_hour=predictions_last_hour,
        predictions_last_day=predictions_last_day,
        average_response_time_ms=avg_response_time,
        error_rate=error_rate,
        uptime_percentage=uptime_percentage,
        models_performance={}  # Could add model-specific metrics
    )


@app.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get current API configuration."""
    return ConfigResponse(**config)


@app.post("/config", response_model=ConfigResponse)
async def update_config(request: ConfigRequest):
    """Update API configuration."""
    # Update configuration
    if request.model_type is not None:
        if request.model_type not in models:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model '{request.model_type}' not available"
            )
        config["default_model"] = request.model_type
    
    if request.confidence_threshold is not None:
        config["confidence_threshold"] = request.confidence_threshold
    
    if request.max_text_length is not None:
        config["max_text_length"] = request.max_text_length
    
    if request.enable_logging is not None:
        config["enable_logging"] = request.enable_logging
    
    if request.cache_predictions is not None:
        config["cache_predictions"] = request.cache_predictions
    
    return ConfigResponse(**config)


# Background tasks
async def process_feedback(feedback_data: Dict[str, Any]):
    """Process feedback in background."""
    # In production, this could:
    # - Store feedback in database
    # - Trigger model retraining
    # - Update model performance metrics
    # - Send notifications for poor performance
    
    logger.info(f"Processing feedback: {feedback_data['id']}")
    
    # Example: Log incorrect predictions
    if feedback_data["predicted_sentiment"] != feedback_data["actual_sentiment"]:
        logger.warning(f"Incorrect prediction for feedback {feedback_data['id']}")


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTPException",
            message=exc.detail,
            request_id=str(uuid.uuid4())
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            details={"exception": str(exc)},
            request_id=str(uuid.uuid4())
        ).dict()
    )


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )













