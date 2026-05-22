# Guide soutenance — 20 minutes

**Laureenda Demeule — Projet OC P7 — Air Paradis**

## Avant la session (5 min)

```bash
cd air-paradis-sentiment && source ../.venv/bin/activate
curl -s https://air-paradis-sentiment-P7.azurewebsites.net/health
export API_BASE_URL=https://air-paradis-sentiment-P7.azurewebsites.net
mlflow ui --port 5000 &
streamlit run streamlit_app/app.py
```

Onglets ouverts : MLflow (localhost:5000), Streamlit (8501), Azure Portal (App Insights), GitHub Actions.

## Déroulé recommandé (~20 min)

| Min | Slide | Action |
|-----|-------|--------|
| 0–2 | 1–2 | Contexte Air Paradis, Sentiment140, split 60/20/20 |
| 2–5 | 3–5 | 3 approches : TF-IDF, CNN-LSTM, DistilBERT (81,5 % F1) |
| 5–7 | 6 | MLflow UI : comparer les runs, montrer le graphique |
| 7–9 | 7–8 | MLOps : GitHub Actions vert, API Azure `/health` |
| 9–12 | 9 | **Demo Streamlit** : tweet négatif → prédiction → feedback incorrect |
| 12–14 | 10 | **App Insights** : trace `misclassified_prediction`, alerte 3/5 min |
| 14–16 | 11 | **Demo live** : tweet positif + négatif |
| 16–18 | 12 | Conclusion : DistilBERT entraîné, logistic déployé sur F1 |
| 18–20 | Q&A | Boucle d'amélioration continue |

## Phrases clés

- « DistilBERT est le meilleur modèle **entraîné** (81,5 % F1 sur 50k). »
- « Sur Azure F1 gratuit, on déploie `tfidf_logistic` — le cahier OC autorise ce fallback mémoire. »
- « Chaque mauvaise prédiction remonte dans App Insights ; alerte à 3 erreurs en 5 minutes. »
- « CI GitHub : 11 tests pytest à chaque push. »

## Tweets de démo

| Tweet | Attendu |
|-------|---------|
| `Air Paradis flight was amazing, best service ever!` | positive |
| `Worst airline ever, delayed again, never flying Air Paradis` | negative |
| `air paradis was awfull` | negative (puis feedback « incorrect » pour App Insights) |

## En cas de problème

| Symptôme | Action |
|----------|--------|
| API Azure down | `bash deployment/deploy_azure.sh` ou API locale `:8000` |
| Streamlit ne répond pas | Vérifier `API_BASE_URL` dans `.env` |
| MLflow vide | `python src/training/pipeline.py --model simple --sample-size 1000` |
| App Insights vide | Vérifier `APPLICATIONINSIGHTS_CONNECTION_STRING` dans `.env` |

## Liens évaluateur

- API : https://air-paradis-sentiment-P7.azurewebsites.net
- GitHub : https://github.com/molly-muffin/P7_OC_AirParadis
- Présentation : `docs/presentation.pdf`
