#!/usr/bin/env bash
# Mirror GitHub Actions test job locally.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export SKIP_MODEL_INIT=1
export PRODUCTION_MODEL=tfidf_logistic
export PYTHONPATH=src

python -m pip install --upgrade pip setuptools wheel -q
pip install -r requirements-ci.txt -q

python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)"

pytest tests/ -v --tb=short
echo "CI repro: all tests passed"
