#!/bin/bash
# Deploy Air Paradis Sentiment API to Azure Web App (F1 free tier, zip deploy)
set -e

RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-air-paradis-rg}"
LOCATION="${AZURE_LOCATION:-westeurope}"
APP_SERVICE_PLAN="${AZURE_APP_PLAN:-air-paradis-plan-f1}"
WEB_APP_NAME="${AZURE_WEBAPP_NAME:-air-paradis-sentiment-laureenda}"
PRODUCTION_MODEL="${AZURE_PRODUCTION_MODEL:-tfidf_logistic}"

if ! command -v az >/dev/null 2>&1; then
  echo "Azure CLI required. From project venv: pip install azure-cli"
  exit 1
fi

if ! bash scripts/check_azure_prereqs.sh; then
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [ -f .env ]; then set -a; source .env; set +a; fi

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

echo "[5/6] Zip package (slim F1 bundle)..."
ZIP="/tmp/air-paradis-deploy.zip"
rm -f "$ZIP"
STAGING="/tmp/air-paradis-staging"
rm -rf "$STAGING"
mkdir -p "$STAGING/models/production"

cp -r src "$STAGING/"
cp deployment/requirements-azure.txt "$STAGING/requirements.txt"
echo "$PRODUCTION_MODEL" > "$STAGING/models/production/production_model.txt"
cp models/production/tfidf_logistic_bundle.pkl "$STAGING/models/production/"

if [ "$PRODUCTION_MODEL" != "tfidf_logistic" ]; then
  echo "WARNING: Non-logistic models may fail on F1 (RAM/build limits). Using $PRODUCTION_MODEL."
  cp -r "models/production/${PRODUCTION_MODEL}" "$STAGING/models/production/" 2>/dev/null || true
fi

(cd "$STAGING" && zip -r "$ZIP" . -x "*.pyc" "*__pycache__*")
echo "Package size: $(du -h "$ZIP" | cut -f1)"

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
