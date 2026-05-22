#!/bin/bash
# Deploy Air Paradis Sentiment API to Azure Web App (F1 free tier, zip deploy)
set -e

RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-air-paradis-rg}"
LOCATION="${AZURE_LOCATION:-westeurope}"
APP_SERVICE_PLAN="${AZURE_APP_PLAN:-air-paradis-plan-f1}"
WEB_APP_NAME="${AZURE_WEBAPP_NAME:-air-paradis-sentiment-laureenda}"
PRODUCTION_MODEL="${AZURE_PRODUCTION_MODEL:-distilbert_finetuned}"

if ! command -v az >/dev/null 2>&1; then
  echo "Azure CLI required. From project venv: pip install azure-cli"
  exit 1
fi

if ! az account show >/dev/null 2>&1; then
  echo "Run: az login"
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "[1/6] Resource group..."
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --output none

echo "[2/6] App Service Plan (F1)..."
az appservice plan create \
  --name "$APP_SERVICE_PLAN" \
  --resource-group "$RESOURCE_GROUP" \
  --sku F1 \
  --is-linux \
  --output none 2>/dev/null || true

echo "[3/6] Web App (Python 3.11)..."
az webapp create \
  --name "$WEB_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --plan "$APP_SERVICE_PLAN" \
  --runtime "PYTHON:3.11" \
  --output none 2>/dev/null || true

echo "[4/6] App settings..."
SETTINGS="SCM_DO_BUILD_DURING_DEPLOYMENT=true"
SETTINGS="$SETTINGS PYTHONPATH=/home/site/wwwroot"
if [ -n "$APPLICATIONINSIGHTS_CONNECTION_STRING" ]; then
  SETTINGS="$SETTINGS APPLICATIONINSIGHTS_CONNECTION_STRING=$APPLICATIONINSIGHTS_CONNECTION_STRING"
fi
# Fallback to logistic on F1 if DistilBERT OOM: export AZURE_PRODUCTION_MODEL=tfidf_logistic
echo "$PRODUCTION_MODEL" > models/production/production_model.txt

az webapp config set \
  --name "$WEB_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --startup-file "python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000" \
  --output none

az webapp config appsettings set \
  --name "$WEB_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --settings $SETTINGS \
  --output none

echo "[5/6] Zip package (code + models/production)..."
ZIP="/tmp/air-paradis-deploy.zip"
rm -f "$ZIP"
zip -r "$ZIP" \
  src/ \
  models/production/ \
  requirements.txt \
  -x "*.pyc" "*__pycache__*" "*.keras" 2>/dev/null || true

# Always include logistic bundle; include BERT only if selected
if [ "$PRODUCTION_MODEL" = "tfidf_logistic" ]; then
  zip -u "$ZIP" models/production/tfidf_logistic_bundle.pkl models/production/production_model.txt
else
  zip -r "$ZIP" models/production/distilbert_finetuned/ models/production/tfidf_logistic_bundle.pkl
fi

echo "[6/6] Zip deploy..."
az webapp deploy \
  --resource-group "$RESOURCE_GROUP" \
  --name "$WEB_APP_NAME" \
  --src-path "$ZIP" \
  --type zip \
  --async true

API_URL="https://${WEB_APP_NAME}.azurewebsites.net"
echo ""
echo "Deployment started (F1 cold start may take 2-5 min)."
echo "API URL: $API_URL"
echo "Health:  $API_URL/health"
echo ""
echo "Update .env: API_BASE_URL=$API_URL"
echo "Test: curl $API_URL/health"
