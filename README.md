# Air Paradis — Analyse de Sentiment (Projet OC P7)

Prototype MLOps pour prédire le sentiment des tweets Air Paradis. Trois approches comparées : modèles simples (TF-IDF), deep learning (CNN-LSTM + embeddings), BERT (DistilBERT).

## Structure

```
air-paradis-sentiment/
├── notebooks/           # EDA + modélisation (MLflow)
├── src/
│   ├── data/            # Chargement et preprocessing
│   ├── models/          # Modèles + predictor production
│   ├── training/        # Pipeline d'entraînement
│   ├── api/             # FastAPI (main.py)
│   └── monitoring/      # Application Insights
├── streamlit_app/       # Interface de test API
├── tests/               # Tests unitaires
├── deployment/          # Docker + Azure
├── models/production/   # Modèles entraînés
├── mlruns/              # Experiments MLflow
├── blog/                # Article MLOps
└── docs/                # Support soutenance
```

## Installation

```bash
cd air-paradis-sentiment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export NLTK_DATA="$(pwd)/data/cache/nltk_data"
mkdir -p data/cache/nltk_data
python -c "import nltk; nltk.download('punkt', download_dir='data/cache/nltk_data'); nltk.download('stopwords', download_dir='data/cache/nltk_data'); nltk.download('punkt_tab', download_dir='data/cache/nltk_data')"
cp .env.example .env
```

## Données

- Brut : `data/raw/training.1600000.processed.noemoticon.csv`
- Nettoyé : `data/processed/cleaned_dataset_sentiment.csv`

## Entraînement

```bash
# Tous les modèles (échantillon 50k par défaut)
python src/training/pipeline.py --sample-size 50000

# Par type
python src/training/pipeline.py --model simple --sample-size 8000
python src/training/pipeline.py --model dl --sample-size 5000
python src/training/pipeline.py --model bert --sample-size 3000
```

## MLflow

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

## API locale

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text":"I love Air Paradis!"}'
```

## Streamlit

```bash
streamlit run streamlit_app/app.py
```

## Tests

```bash
pytest tests/ -v
```

## Déploiement Azure (Web App F1)

**API production :** https://air-paradis-sentiment-P7.azurewebsites.net

- Health : https://air-paradis-sentiment-P7.azurewebsites.net/health
- Docs : https://air-paradis-sentiment-P7.azurewebsites.net/docs

**Code source :** https://github.com/molly-muffin/P7_OC_AirParadis

```bash
az login
./deployment/deploy_azure.sh
```

Variables requises : `APPLICATIONINSIGHTS_CONNECTION_STRING`, secrets GitHub `AZURE_CREDENTIALS`, `AZURE_WEBAPP_NAME`.

CI/CD : le job `test` (pytest) tourne à chaque push. Le job `deploy` est désactivé par défaut ; activer avec la variable repo `ENABLE_AZURE_DEPLOY=true` après configuration des secrets.

## Résultats (échantillon 50k tweets, split 60/20/20)

| Modèle | Val F1 | Test Accuracy | Test F1 |
|--------|--------|--------------|---------|
| TF-IDF + LogReg | 76.9% | 76.4% | 76.4% |
| TF-IDF + Random Forest | 72.5% | 70.9% | 72.5% |
| CNN-LSTM + Word2Vec | 64.9% | 67.7% | 65.7% |
| CNN-LSTM + GloVe | 68.5% | 71.4% | 69.4% |
| **DistilBERT** | **80.2%** | **81.5%** | **81.5%** |

Modèle production sélectionné : **DistilBERT**. Fallback Azure F1 : `tfidf_logistic`.

## Livrables OC

| N° | Contenu | Emplacement |
|----|---------|-------------|
| 1 | API cloud | Azure Web App URL (README) |
| 2 | Scripts + notebooks | `src/training/`, `notebooks/` |
| 3 | Code versionné | GitHub |
| 4 | Interface test | `streamlit_app/app.py` |
| 5 | Blog MLOps | `blog/article_mlops_sentiment.md` |
| 6 | Présentation | `docs/presentation.md` |

Nommage zip : `Demeule_Laureenda_N_livrable_mmaaaa`
