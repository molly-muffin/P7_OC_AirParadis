# Production model artifacts

Large weights are excluded from Git (see `.gitignore`). Regenerate after clone:

```bash
python src/training/pipeline.py --sample-size 50000
```

For Azure F1 deploy without GPU, fallback:

```bash
echo "tfidf_logistic" > production_model.txt
```

Included in repo: `model_comparison.json`, `production_model.txt`, tokenizer configs, `tfidf_logistic_bundle.pkl` (fallback).
