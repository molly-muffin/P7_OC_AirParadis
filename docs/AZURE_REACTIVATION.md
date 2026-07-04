# Réactivation Azure — Avant la soutenance

## Problème actuel

La subscription Azure (`a9e0916d-25c3-4477-88b0-8e243ddc82dc`) est **désactivée**.
L'API retourne HTTP 403 "Web App is stopped".

Cette erreur survient quand :
- L'abonnement Azure gratuit (12 mois) a expiré, ou
- Les crédits gratuits sont épuisés, ou
- La subscription a été suspendue pour inactivité

## Étapes pour réactiver (5-10 min)

1. Aller sur https://portal.azure.com
2. Menu gauche → **Abonnements** (Subscriptions)
3. Cliquer sur l'abonnement `Azure subscription 1`
4. Si le statut est "Désactivé" → bouton **Réactiver** (peut nécessiter une carte bancaire)
5. Attendre 5-10 min que la subscription soit active
6. Aller sur **App Services** → `air-paradis-sentiment-p7` → bouton **Démarrer**
7. Vérifier :

```bash
curl https://air-paradis-sentiment-P7.azurewebsites.net/health
```

## Après réactivation — redéployer avec CNN-LSTM

L'API doit exposer le **Modèle sur mesure avancé** (CNN-LSTM GloVe) pour la conformité OC.

```bash
cd air-paradis-sentiment && source ../.venv/bin/activate
az login
export AZURE_WEBAPP_NAME=air-paradis-sentiment-P7
export AZURE_PRODUCTION_MODEL=cnn_lstm_glove
bash deployment/deploy_azure.sh
```

Attendre 3-5 min (cold start F1), puis vérifier :

```bash
curl -s https://air-paradis-sentiment-P7.azurewebsites.net/health | python3 -m json.tool
# Attendu : "model": "cnn_lstm_glove", "model_loaded": true
```

## Si le déploiement CNN-LSTM échoue (TF trop lourd)

Fallback acceptable à l'oral :

```bash
export AZURE_PRODUCTION_MODEL=tfidf_logistic
bash deployment/deploy_azure.sh
```

Justification à préparer :
"DistilBERT est le meilleur modèle entraîné (81,5 % F1). Sur Azure F1 (512 Mo RAM),
TensorFlow ne peut pas s'installer. Le CNN-LSTM est déployé si les ressources le permettent,
sinon la régression logistique sert de fallback ultra-léger (76,4 % F1)."

## Alternative — Plan B sans Azure (démo locale)

Si la subscription ne peut pas être réactivée avant la soutenance :

```bash
# Lancer l'API en local
cd air-paradis-sentiment && source ../.venv/bin/activate
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Configurer Streamlit sur l'API locale
echo "API_BASE_URL=http://localhost:8000" >> .env
streamlit run streamlit_app/app.py
```

Dans ce cas, montrer le code de déploiement Azure et les logs de déploiement précédents
comme preuve que le déploiement cloud a fonctionné.
