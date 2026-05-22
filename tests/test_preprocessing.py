"""Tests for text preprocessing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.preprocessor import TextPreprocessor


def test_clean_text_removes_url():
    preprocessor = TextPreprocessor(cache_dir="data/cache/test")
    text = "Check this http://example.com now"
    cleaned = preprocessor.clean_text(text)
    assert "http" not in cleaned
    assert "example.com" not in cleaned


def test_clean_text_lowercase():
    preprocessor = TextPreprocessor(cache_dir="data/cache/test")
    assert preprocessor.clean_text("HELLO World") == "hello world"


def test_clean_text_removes_mention():
    preprocessor = TextPreprocessor(cache_dir="data/cache/test")
    cleaned = preprocessor.clean_text("@user hello world")
    assert "@" not in cleaned
    assert "user" in cleaned.lower() or "USER" in cleaned
