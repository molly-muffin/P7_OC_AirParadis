#!/bin/bash
# Diagnose Azure CLI auth and subscription state before deploy
set -e
cd "$(dirname "$0")/.."

DEBUG_LOG="${DEBUG_LOG:-/Users/laureendademeule/Documents/Projets/P7/.cursor/debug-448d12.log}"
RUN_ID="${RUN_ID:-pre-fix}"

_log() {
  local hypothesis_id="$1" location="$2" message="$3" data="$4"
  # #region agent log
  printf '{"sessionId":"448d12","runId":"%s","hypothesisId":"%s","location":"%s","message":"%s","data":%s,"timestamp":%s}\n' \
    "$RUN_ID" "$hypothesis_id" "$location" "$message" "$data" "$(date +%s000)" >> "$DEBUG_LOG"
  # #endregion
}

if ! command -v az >/dev/null 2>&1; then
  _log "H4" "check_azure_prereqs.sh" "az_cli_missing" '{"az_found":false}'
  echo "ERROR: Azure CLI not installed."
  echo "Fix: source ../.venv/bin/activate  (azure-cli is in the project venv)"
  exit 1
fi
_log "H4" "check_azure_prereqs.sh" "az_cli_ok" '{"az_found":true}'

USER_NAME=$(az account show --query user.name -o tsv 2>/dev/null || true)
SUB_COUNT=$(az account list --query "length(@)" -o tsv 2>/dev/null || echo "0")
TENANT_COUNT=$(az account tenant list --query "length(@)" -o tsv 2>/dev/null || echo "0")

_log "H1" "check_azure_prereqs.sh" "account_state" \
  "{\"user\":\"${USER_NAME:-none}\",\"subscription_count\":${SUB_COUNT:-0},\"tenant_count\":${TENANT_COUNT:-0}}"

if [ -z "$USER_NAME" ]; then
  _log "H2" "check_azure_prereqs.sh" "not_logged_in" '{"logged_in":false}'
  echo "ERROR: Not logged in to Azure."
  echo "Fix: az login"
  exit 1
fi

if [ "${SUB_COUNT:-0}" = "0" ] || [ "$SUB_COUNT" = "None" ]; then
  _log "H1" "check_azure_prereqs.sh" "no_subscription" \
    "{\"logged_in\":true,\"user\":\"$USER_NAME\",\"subscription_count\":0}"
  echo "ERROR: Logged in as $USER_NAME but no Azure subscription is linked to this account."
  echo ""
  echo "This is not a login failure. Azure CLI authenticated you, but there is nothing to deploy to."
  echo ""
  echo "Fix (pick one):"
  echo "  1. Create a free Azure account: https://azure.microsoft.com/free/"
  echo "     (requires credit card for identity verification; F1 tier stays free)"
  echo "  2. Activate OpenClassrooms / Azure for Students if you have student access"
  echo "  3. Log in with the Microsoft account that owns your Azure subscription:"
  echo "       az logout && az login"
  echo "  4. If your subscription is under another tenant (work/school):"
  echo "       az login --tenant YOUR_TENANT_ID"
  exit 2
fi

SUB_NAME=$(az account show --query name -o tsv 2>/dev/null)
SUB_ID=$(az account show --query id -o tsv 2>/dev/null)
_log "H3" "check_azure_prereqs.sh" "ready" \
  "{\"logged_in\":true,\"user\":\"$USER_NAME\",\"subscription\":\"$SUB_NAME\",\"subscription_id\":\"$SUB_ID\"}"

echo "Azure OK: $USER_NAME | subscription: $SUB_NAME"
exit 0
