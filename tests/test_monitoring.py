"""Tests for Application Insights monitoring."""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from monitoring import app_insights


def test_track_misclassification_without_connection_string():
    with patch.dict("os.environ", {}, clear=True):
        result = app_insights.track_misclassification(
            tweet_text="test tweet",
            predicted_sentiment="positive",
            user_feedback="negative",
            confidence=0.8,
        )
    assert result is True


def test_alert_threshold():
    app_insights._misclassification_buffer = [
        datetime.utcnow() - timedelta(minutes=1) for _ in range(3)
    ]
    assert app_insights.check_alert_threshold(threshold=3) is True

    app_insights._misclassification_buffer = []
    assert app_insights.check_alert_threshold(threshold=3) is False
