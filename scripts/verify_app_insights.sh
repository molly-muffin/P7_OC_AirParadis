#!/bin/bash
# Verify Application Insights configuration
set -e
cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

if [ -z "$APPLICATIONINSIGHTS_CONNECTION_STRING" ]; then
  echo "MISSING: APPLICATIONINSIGHTS_CONNECTION_STRING"
  echo "Create Application Insights in Azure Portal and add connection string to .env"
  echo "See docs/AZURE_SETUP.md"
  exit 1
fi

echo "Application Insights connection string is configured."
echo "Test: run Streamlit, submit incorrect feedback, check Azure Portal > Application Insights > Logs"

../.venv/bin/python -c "
import os, sys
sys.path.insert(0, 'src')
from monitoring.app_insights import track_misclassification
track_misclassification('test tweet', 'positive', 'negative', 0.5)
print('Test trace sent (or logged locally if SDK unavailable)')
"
