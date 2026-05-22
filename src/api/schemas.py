"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class SentimentLabel(str, Enum):
    """Sentiment label enumeration."""
    NEGATIVE = "negative"
    POSITIVE = "positive"


class PredictionRequest(BaseModel):
    """Request schema for sentiment prediction."""
    text: str = Field(..., min_length=1, max_length=1000, description="Text to analyze")
    model_type: Optional[str] = Field(default="best", description="Model to use for prediction")
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or only whitespace')
        return v.strip()


class BatchPredictionRequest(BaseModel):
    """Request schema for batch sentiment prediction."""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to analyze")
    model_type: Optional[str] = Field(default="best", description="Model to use for prediction")
    
    @validator('texts')
    def validate_texts(cls, v):
        validated_texts = []
        for text in v:
            if not text or not text.strip():
                raise ValueError('All texts must be non-empty')
            validated_texts.append(text.strip())
        return validated_texts


class PredictionResponse(BaseModel):
    """Response schema for sentiment prediction."""
    text: str = Field(..., description="Original text")
    sentiment: SentimentLabel = Field(..., description="Predicted sentiment")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")
    model_used: str = Field(..., description="Model used for prediction")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Prediction timestamp")


class BatchPredictionResponse(BaseModel):
    """Response schema for batch sentiment prediction."""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_count: int = Field(..., description="Total number of predictions")
    model_used: str = Field(..., description="Model used for predictions")
    total_processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Batch prediction timestamp")


class FeedbackRequest(BaseModel):
    """Request schema for prediction feedback."""
    text: str = Field(..., description="Original text")
    predicted_sentiment: SentimentLabel = Field(..., description="Model prediction")
    actual_sentiment: SentimentLabel = Field(..., description="Actual sentiment")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    model_used: str = Field(..., description="Model that made the prediction")
    user_id: Optional[str] = Field(default=None, description="Optional user identifier")
    comments: Optional[str] = Field(default=None, max_length=500, description="Optional feedback comments")


class FeedbackResponse(BaseModel):
    """Response schema for feedback submission."""
    message: str = Field(..., description="Confirmation message")
    feedback_id: str = Field(..., description="Unique feedback identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Feedback submission timestamp")


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    version: str = Field(..., description="API version")
    models_loaded: List[str] = Field(..., description="List of loaded models")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class ModelInfo(BaseModel):
    """Schema for model information."""
    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type")
    version: str = Field(..., description="Model version")
    accuracy: Optional[float] = Field(default=None, description="Model accuracy")
    f1_score: Optional[float] = Field(default=None, description="Model F1 score")
    training_date: Optional[datetime] = Field(default=None, description="Model training date")
    parameters: Optional[int] = Field(default=None, description="Number of model parameters")
    status: str = Field(..., description="Model status (loaded/not_loaded)")


class ModelInfoResponse(BaseModel):
    """Response schema for model information."""
    current_model: ModelInfo = Field(..., description="Currently active model")
    available_models: List[ModelInfo] = Field(..., description="List of available models")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Information retrieval timestamp")


class ErrorResponse(BaseModel):
    """Response schema for API errors."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(default=None, description="Request identifier for tracking")


class MetricsResponse(BaseModel):
    """Response schema for API metrics."""
    total_predictions: int = Field(..., description="Total number of predictions made")
    predictions_last_hour: int = Field(..., description="Predictions in the last hour")
    predictions_last_day: int = Field(..., description="Predictions in the last day")
    average_response_time_ms: float = Field(..., description="Average response time in milliseconds")
    error_rate: float = Field(..., ge=0.0, le=1.0, description="Error rate (0-1)")
    uptime_percentage: float = Field(..., ge=0.0, le=100.0, description="Uptime percentage")
    models_performance: Dict[str, Dict[str, float]] = Field(..., description="Performance metrics by model")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Metrics timestamp")


class ConfigRequest(BaseModel):
    """Request schema for configuration updates."""
    model_type: Optional[str] = Field(default=None, description="Default model type to use")
    confidence_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence threshold")
    max_text_length: Optional[int] = Field(default=None, ge=1, le=5000, description="Maximum text length")
    enable_logging: Optional[bool] = Field(default=None, description="Enable detailed logging")
    cache_predictions: Optional[bool] = Field(default=None, description="Enable prediction caching")


class ConfigResponse(BaseModel):
    """Response schema for configuration."""
    model_type: str = Field(..., description="Current default model type")
    confidence_threshold: float = Field(..., description="Current confidence threshold")
    max_text_length: int = Field(..., description="Maximum allowed text length")
    enable_logging: bool = Field(..., description="Logging status")
    cache_predictions: bool = Field(..., description="Prediction caching status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Configuration timestamp")


# Custom validators
def validate_sentiment_score(v: float) -> float:
    """Validate sentiment confidence score."""
    if not 0.0 <= v <= 1.0:
        raise ValueError('Confidence score must be between 0.0 and 1.0')
    return round(v, 4)


def validate_model_type(v: str) -> str:
    """Validate model type."""
    allowed_types = ['logistic', 'lstm', 'cnn', 'hybrid', 'bert', 'distilbert', 'best']
    if v.lower() not in allowed_types:
        raise ValueError(f'Model type must be one of: {", ".join(allowed_types)}')
    return v.lower()


# Apply validators to relevant fields
PredictionResponse.model_validate = validator('confidence', allow_reuse=True)(validate_sentiment_score)
BatchPredictionRequest.model_validate = validator('model_type', allow_reuse=True)(validate_model_type)
PredictionRequest.model_validate = validator('model_type', allow_reuse=True)(validate_model_type)













