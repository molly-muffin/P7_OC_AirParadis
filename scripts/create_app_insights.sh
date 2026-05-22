#!/bin/bash
# Create Application Insights resource and write connection string to .env
set -e
cd "$(dirname "$0")/.."

RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-air-paradis-rg}"
LOCATION="${AZURE_LOCATION:-westeurope}"
APP_INSIGHTS_NAME="${AZURE_APP_INSIGHTS_NAME:-air-paradis-insights}"
DEBUG_LOG="${DEBUG_LOG:-/Users/laureendademeule/Documents/Projets/P7/.cursor/debug-448d12.log}"
RUN_ID="${RUN_ID:-deploy}"

_log() {
  local hypothesis_id="$1" location="$2" message="$3" data="$4"
  # #region agent log
  printf '{"sessionId":"448d12","runId":"%s","hypothesisId":"%s","location":"%s","message":"%s","data":%s,"timestamp":%s}\n' \
    "$RUN_ID" "$hypothesis_id" "$location" "$message" "$data" "$(date +%s000)" >> "$DEBUG_LOG"
  # #endregion
}

if ! command -v az >/dev/null 2>&1; then
  echo "Azure CLI not found. Install: pip install azure-cli (or brew install azure-cli)"
  exit 1
fi

if ! bash scripts/check_azure_prereqs.sh; then
  exit 1
fi

echo "Creating resource group (if needed)..."
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --output none

for NS in Microsoft.Insights Microsoft.OperationalInsights; do
  STATE=$(az provider show -n "$NS" --query registrationState -o tsv 2>/dev/null || echo "NotRegistered")
  _log "H1" "create_app_insights.sh" "provider_state" "{\"namespace\":\"$NS\",\"state\":\"$STATE\"}"
  if [ "$STATE" != "Registered" ]; then
    echo "Registering Azure provider: $NS (first-time setup, ~1 min)..."
    az provider register --namespace "$NS" --wait --output none
    _log "H1" "create_app_insights.sh" "provider_registered" "{\"namespace\":\"$NS\"}"
  fi
done

echo "Creating Application Insights: $APP_INSIGHTS_NAME"
if az monitor app-insights component show \
  --app "$APP_INSIGHTS_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query name -o tsv >/dev/null 2>&1; then
  echo "Application Insights already exists, fetching connection string..."
  _log "H2" "create_app_insights.sh" "insights_exists" "{\"name\":\"$APP_INSIGHTS_NAME\"}"
else
  CREATE_OUT=$(az monitor app-insights component create \
    --app "$APP_INSIGHTS_NAME" \
    --location "$LOCATION" \
    --resource-group "$RESOURCE_GROUP" \
    --application-type web \
    --query connectionString -o tsv 2>&1) || {
    _log "H1" "create_app_insights.sh" "create_failed" "{\"error\":\"${CREATE_OUT//\"/\\\"}\"}"
    echo "ERROR: Failed to create Application Insights."
    echo "$CREATE_OUT"
    exit 1
  }
  _log "H1" "create_app_insights.sh" "create_ok" "{\"name\":\"$APP_INSIGHTS_NAME\"}"
fi

CONN=$(az monitor app-insights component show \
  --app "$APP_INSIGHTS_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query connectionString -o tsv)

if [ -z "$CONN" ]; then
  _log "H3" "create_app_insights.sh" "conn_missing" "{\"name\":\"$APP_INSIGHTS_NAME\"}"
  echo "Failed to retrieve connection string"
  exit 1
fi

_log "H3" "create_app_insights.sh" "conn_ok" "{\"has_connection_string\":true}"

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
