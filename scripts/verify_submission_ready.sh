#!/usr/bin/env bash
# Pre-soutenance and pre-submission checks.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== Azure API ==="
curl -sf -m 15 "https://air-paradis-sentiment-P7.azurewebsites.net/health" | python -m json.tool

echo ""
echo "=== Pytest ==="
export SKIP_MODEL_INIT=1 PRODUCTION_MODEL=tfidf_logistic PYTHONPATH=src
pytest tests/ -q

echo ""
echo "=== Livrables ==="
GLOBAL="$ROOT/livrables/Realisez_une_analyse_de_sentiments_Demeule_Laureenda.zip"
test -f "$GLOBAL" && echo "OK: $GLOBAL ($(du -h "$GLOBAL" | cut -f1))"

echo ""
echo "=== Presentation ==="
test -f "$ROOT/docs/presentation.pdf" && echo "OK: docs/presentation.pdf"
test -f "$ROOT/docs/presentation.pptx" && echo "OK: docs/presentation.pptx"

echo ""
echo "=== Screenshots ==="
for f in mlflow_ui_runs.png github_actions_ci.png streamlit_demo.png app_insights_traces.png azure_api_health.png; do
  test -f "$ROOT/docs/screenshots/$f" && echo "OK: docs/screenshots/$f"
done

echo ""
echo "All checks passed. Ready for soutenance and OC upload."
echo "Upload: $GLOBAL"
echo "Guide: docs/SUBMISSION_OC.md | docs/SOUTENANCE_GUIDE.md"
