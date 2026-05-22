# Azure Setup Guide (manual steps)

## 1. Application Insights

1. Go to [portal.azure.com](https://portal.azure.com)
2. Create resource > **Application Insights**
3. Name: `air-paradis-insights`
4. Resource group: `air-paradis-rg` (create if needed)
5. Region: West Europe
6. After creation: **Properties** > copy **Connection String**
7. Add to `.env`:
   ```
   APPLICATIONINSIGHTS_CONNECTION_STRING=InstrumentationKey=...;IngestionEndpoint=...
   ```

## 2. Deploy API (Web App F1)

```bash
az login
az account set --subscription "YOUR_SUBSCRIPTION"

export AZURE_WEBAPP_NAME="air-paradis-sentiment-laureenda"   # must be globally unique
export APPLICATIONINSIGHTS_CONNECTION_STRING="..."

cd air-paradis-sentiment
source ../.venv/bin/activate
az login
chmod +x deployment/deploy_azure.sh
./deployment/deploy_azure.sh
```

Or: `bash scripts/finish_deployment.sh` (after `az login` and optional `gh auth login`)

## 3. Test cloud API

```bash
curl https://YOUR_APP.azurewebsites.net/health
curl -X POST https://YOUR_APP.azurewebsites.net/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"I love Air Paradis flights!"}'
```

## 4. Streamlit against cloud

Update `.env`:
```
API_BASE_URL=https://YOUR_APP.azurewebsites.net
```

```bash
streamlit run streamlit_app/app.py
```

## 5. GitHub Actions secrets

In GitHub repo Settings > Secrets > Actions:

| Secret | Value |
|--------|-------|
| `AZURE_CREDENTIALS` | JSON from `az ad sp create-for-rbac --sdk-auth` |
| `AZURE_WEBAPP_NAME` | Your web app name |

Create service principal:
```bash
az ad sp create-for-rbac --name "air-paradis-github" \
  --role contributor \
  --scopes /subscriptions/SUBSCRIPTION_ID/resourceGroups/air-paradis-rg \
  --sdk-auth
```

## Fallback: logistic model on F1

If DistilBERT fails (OOM / timeout on free tier):

```bash
echo "tfidf_logistic" > models/production/production_model.txt
# Redeploy with logistic bundle only (~few MB)
```
