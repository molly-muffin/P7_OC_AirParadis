"""Pytest configuration for CI and local runs."""

import os

# Must be set before test modules import api.main
os.environ.setdefault("SKIP_MODEL_INIT", "1")
os.environ.setdefault("PRODUCTION_MODEL", "tfidf_logistic")
