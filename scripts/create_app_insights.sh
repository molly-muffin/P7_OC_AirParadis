#!/bin/bash
# Create Application Insights resource and write connection string to .env
set -e
cd "$(dirname "$0")/.."

RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-air-paradis-rg}"
LOCATION="${AZURE_LOCATION:-westeurope}"
APP_INSIGHTS_NAME="${AZURE_APP_INSIGHTS_NAME:-air-paradis-insights}"

if ! command -v az >/dev/null 2>&1; then
  echo "Azure CLI not found. Install: pip install azure-cli (or brew install azure-cli)"
  exit 1
fi

if ! az account show >/dev/null 2>&1; then
  echo "Run: az login"
  exit 1
fi

echo "Creating resource group (if needed)..."
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --output none

echo "Creating Application Insights: $APP_INSIGHTS_NAME"
az monitor app-insights component create \
  --app "$APP_INSIGHTS_NAME" \
  --location "$LOCATION" \
  --resource-group "$RESOURCE_GROUP" \
  --application-type web \
  --output none 2>/dev/null || true

CONN=$(az monitor app-insights component show \
  --app "$APP_INSIGHTS_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query connectionString -o tsv)

if [ -z "$CONN" ]; then
  echo "Failed to retrieve connection string"
  exit 1
fi

touch .env
if grep -q "^APPLICATIONINSIGHTS_CONNECTION_STRING=" .env 2>/dev/null; then
  sed -i.bak "s|^APPLICATIONINSIGHTS_CONNECTION_STRING=.*|APPLICATIONINSIGHTS_CONNECTION_STRING=$CONN|" .env
  rm -f .env.bak
else
  echo "APPLICATIONINSIGHTS_CONNECTION_STRING=$CONN" >> .env
fi

export APPLICATIONINSIGHTS_CONNECTION_STRING="$CONN"
echo "Connection string saved to .env"
bash scripts/verify_app_insights.sh
