#!/usr/bin/env bash
# Mirror GitHub Actions test job locally and write debug NDJSON.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEBUG_LOG="${DEBUG_LOG:-$ROOT/../.cursor/debug-448d12.log}"
RUN_ID="${RUN_ID:-local-ci-repro}"

log_json() {
  python - <<PY
import json, time, os
payload = {
    "sessionId": "448d12",
    "runId": os.environ.get("RUN_ID", "local-ci-repro"),
    "hypothesisId": "$1",
    "location": "scripts/repro_ci.sh",
    "message": "$2",
    "data": json.loads('''$3'''),
    "timestamp": int(time.time() * 1000),
}
path = os.environ["DEBUG_LOG"]
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(payload) + "\n")
PY
}

export SKIP_MODEL_INIT=1
export PRODUCTION_MODEL=tfidf_logistic
export DEBUG_LOG
export RUN_ID

cd "$ROOT"
log_json "H1" "env" "{\"SKIP_MODEL_INIT\":\"$SKIP_MODEL_INIT\",\"PRODUCTION_MODEL\":\"$PRODUCTION_MODEL\",\"python\":\"$(python --version 2>&1)\"}"

python -m pip install --upgrade pip setuptools wheel -q
pip install -r requirements-ci.txt -q
log_json "H2" "deps_installed" "{\"requirements\":\"requirements-ci.txt\"}"

python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)"
log_json "H3" "nltk_ready" "{}"

if pytest tests/ -v --tb=short; then
  log_json "H4" "pytest_passed" "{\"exit_code\":0}"
  echo "CI repro: all tests passed"
else
  code=$?
  log_json "H4" "pytest_failed" "{\"exit_code\":$code}"
  exit "$code"
fi
