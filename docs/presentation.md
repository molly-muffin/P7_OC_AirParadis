# Support de présentation — Air Paradis Sentiment Analysis

**Laureenda Demeule — Projet OC P7 — Soutenance**

---

## Slide 1 — Contexte

- Client : Air Paradis (compagnie aérienne)
- Besoin : prédire le sentiment d'un tweet (positif / négatif)
- Données : Sentiment140 (1,6M tweets, open source)
- Livrable : prototype MLOps déployé sur le cloud

---

## Slide 2 — Méthodologie globale

1. EDA + nettoyage (`01_Analyse_Exploratoire.ipynb`)
2. Trois approches de modélisation
3. Tracking MLflow
4. Sélection du meilleur modèle
5. Déploiement API Azure + monitoring App Insights

Split des données : **60 % train / 20 % validation / 20 % test**

---

## Slide 3 — Approche 1 : Modèles simples

- TF-IDF (5000 features)
- Régression logistique : ~79 % accuracy (dataset complet)
- Random Forest : F1 test ~74 %

Avantages : rapide, léger, interprétable.

---

## Slide 4 — Approche 2 : Deep Learning

- CNN-LSTM hybride
- Embeddings : Word2Vec vs GloVe
- Early stopping sur validation

Performances modérées sur petit échantillon ; s'améliore avec plus de données.

---

## Slide 5 — Approche 3 : BERT

- DistilBERT fine-tuned
- 2 epochs, échantillon stratifié
- **Meilleur modèle : 76,5 % test accuracy, 75,8 % F1**

Sélectionné pour la production.

---

## Slide 6 — Comparaison MLflow

| Modèle | Test F1 |
|--------|---------|
| LogReg | 76,4 % |
| Random Forest | 72,5 % |
| CNN-LSTM W2V | 65,7 % |
| CNN-LSTM GloVe | 69,4 % |
| **DistilBERT** | **81,5 %** |

![Comparaison MLflow](../screenshots/mlflow_model_comparison.png)

*[Capture écran MLflow UI — voir aussi `docs/screenshots/pytest_ci_output.txt` pour CI locale]*

---

## Slide 7 — MLOps : principes

- **Tracking** : MLflow (params, metrics, artifacts)
- **Versioning** : Git + GitHub
- **Tests** : pytest (11 tests)
- **CI/CD** : GitHub Actions
- **Monitoring** : Azure Application Insights

---

## Slide 8 — Mise en production

- API FastAPI : `/predict`, `/health`, `/feedback`
- Déploiement : Azure Web App F1
- Docker : `deployment/Dockerfile`
- Fallback : logistic regression si limites F1

*[Capture écran API cloud + curl]*

---

## Slide 9 — Interface Streamlit

- Saisie tweet → appel API → prédiction
- Validation utilisateur (Oui / Non)
- Trace App Insights si erreur
- Alerte : 3 erreurs / 5 min

*[Capture écran Streamlit]*

---

## Slide 10 — Suivi production

- Traces : texte, prédiction, timestamp
- Alertes configurées dans App Insights
- Boucle d'amélioration : collecte → analyse → réentraînement → MLflow compare → promotion

*[Capture écran Azure Application Insights]*

---

## Slide 11 — Démonstration live

1. Tweet positif → prédiction positive
2. Tweet négatif → prédiction négative
3. Feedback incorrect → trace App Insights

---

## Slide 12 — Conclusion

- DistilBERT = meilleur modèle pour ce cas
- MLOps opérationnel (tracking, CI, monitoring)
- Piste d'évolution : plus de données, active learning via feedback

---

## Annexes pour l'évaluateur

- Repo : `air-paradis-sentiment/`
- Notebooks : `notebooks/`
- Blog : `blog/article_mlops_sentiment.md`
- Commandes : voir `README.md`
