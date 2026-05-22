"""Pytest configuration for CI and local runs."""

import json
import os
import time
from pathlib import Path

# Must be set before test modules import api.main
os.environ.setdefault("SKIP_MODEL_INIT", "1")
os.environ.setdefault("PRODUCTION_MODEL", "tfidf_logistic")

# #region agent log
def _debug_log(hypothesis_id: str, message: str, data: dict) -> None:
    log_path = Path(
        os.getenv(
            "DEBUG_LOG",
            "/Users/laureendademeule/Documents/Projets/P7/.cursor/debug-448d12.log",
        )
    )
    try:
        payload = {
            "sessionId": "448d12",
            "runId": os.getenv("RUN_ID", "pytest"),
            "hypothesisId": hypothesis_id,
            "location": "tests/conftest.py",
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
    except OSError:
        pass


_debug_log(
    "H1",
    "conftest_env",
    {
        "SKIP_MODEL_INIT": os.environ.get("SKIP_MODEL_INIT"),
        "PRODUCTION_MODEL": os.environ.get("PRODUCTION_MODEL"),
    },
)
# #endregion

