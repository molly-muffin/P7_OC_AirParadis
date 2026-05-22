#!/bin/bash
# Run after: az login && gh auth login
set -e
cd "$(dirname "$0")/.."
source ../.venv/bin/activate 2>/dev/null || true

echo "=== Application Insights ==="
bash scripts/create_app_insights.sh

echo "=== Azure Web App deploy ==="
export AZURE_WEBAPP_NAME="${AZURE_WEBAPP_NAME:-air-paradis-sentiment-laureenda}"
bash deployment/deploy_azure.sh

echo "=== Update Streamlit .env ==="
URL="https://${AZURE_WEBAPP_NAME}.azurewebsites.net"
if grep -q "^API_BASE_URL=" .env 2>/dev/null; then
  sed -i.bak "s|^API_BASE_URL=.*|API_BASE_URL=$URL|" .env && rm -f .env.bak
else
  echo "API_BASE_URL=$URL" >> .env
fi
echo "Set API_BASE_URL=$URL"

echo "=== GitHub (optional) ==="
if [ -n "$GITHUB_REPO_URL" ]; then
  bash scripts/setup_github.sh "$GITHUB_REPO_URL"
  git push -u origin main
fi

echo "Done. Test: curl $URL/health"
