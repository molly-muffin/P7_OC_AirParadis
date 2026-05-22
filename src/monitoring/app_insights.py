"""Azure Application Insights telemetry for misclassified predictions."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_misclassification_buffer: List[datetime] = []


def _get_connection_string() -> Optional[str]:
    return os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")


def track_misclassification(
    tweet_text: str,
    predicted_sentiment: str,
    user_feedback: str,
    confidence: float = 0.0,
) -> bool:
    """
    Send a trace to Application Insights when user rejects a prediction.

    Returns True if telemetry was sent (or logged locally as fallback).
    """
    global _misclassification_buffer

    payload: Dict[str, Any] = {
        "tweet_text": tweet_text[:500],
        "predicted_sentiment": predicted_sentiment,
        "user_feedback": user_feedback,
        "confidence": confidence,
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": "misclassified_prediction",
    }

    conn = _get_connection_string()
    if conn:
        try:
            from opentelemetry import trace
            from azure.monitor.opentelemetry import configure_azure_monitor

            configure_azure_monitor(connection_string=conn)
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span("misclassified_prediction") as span:
                for key, value in payload.items():
                    span.set_attribute(key, str(value))
            logger.info("Sent misclassification trace to Application Insights")
        except ImportError:
            logger.warning("azure-monitor-opentelemetry not installed; logging locally")
            logger.warning("MISCLASSIFICATION: %s", payload)
        except Exception as exc:
            logger.error("Failed to send telemetry: %s", exc)
            logger.warning("MISCLASSIFICATION (fallback): %s", payload)
    else:
        logger.warning("No APPLICATIONINSIGHTS_CONNECTION_STRING; local log only")
        logger.warning("MISCLASSIFICATION: %s", payload)

    _misclassification_buffer.append(datetime.utcnow())
    _misclassification_buffer = [
        ts for ts in _misclassification_buffer if ts > datetime.utcnow() - timedelta(minutes=5)
    ]
    return True


def check_alert_threshold(threshold: int = 3, window_minutes: int = 5) -> bool:
    """Return True if misclassification count exceeds threshold in the time window."""
    global _misclassification_buffer
    cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
    _misclassification_buffer = [ts for ts in _misclassification_buffer if ts > cutoff]
    if len(_misclassification_buffer) >= threshold:
        logger.critical(
            "ALERT: %d misclassified predictions in %d minutes",
            len(_misclassification_buffer),
            window_minutes,
        )
        conn = _get_connection_string()
        if conn:
            try:
                from opentelemetry import trace
                from azure.monitor.opentelemetry import configure_azure_monitor

                configure_azure_monitor(connection_string=conn)
                tracer = trace.get_tracer(__name__)
                with tracer.start_as_current_span("misclassification_alert") as span:
                    span.set_attribute("count", len(_misclassification_buffer))
                    span.set_attribute("window_minutes", window_minutes)
            except Exception:
                pass
        return True
    return False


def get_misclassification_count() -> int:
    cutoff = datetime.utcnow() - timedelta(minutes=5)
    return len([ts for ts in _misclassification_buffer if ts > cutoff])
