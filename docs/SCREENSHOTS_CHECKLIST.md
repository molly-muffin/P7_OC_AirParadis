# Screenshots checklist for soutenance and blog

Capture these after Azure deploy and GitHub push:

## MLflow
- [ ] `mlflow ui --port 5000` — experiment list with all runs
- [ ] Comparison of metrics (val_accuracy, test_f1) per model
- [ ] One run detail page with parameters and artifacts

## API
- [ ] `curl /health` response (local or Azure)
- [ ] `curl /predict` with sample tweet
- [ ] Browser or Swagger if enabled

## Streamlit
- [ ] Tweet input + prediction display
- [ ] Feedback "Non, incorrecte" flow
- [ ] Alert message after 3 misclassifications in 5 min

## Azure Application Insights
- [ ] Transaction search showing `misclassified_prediction` traces
- [ ] Alert log entry (if configured)

## GitHub
- [ ] Repository main page with commits
- [ ] Actions tab — green CI run (pytest job)

## Azure Portal
- [ ] Web App overview with URL
- [ ] Application Insights resource linked

Save images in `docs/screenshots/` and reference in blog + presentation.

**Generated locally (ready for blog/slides):**
- `mlflow_model_comparison.png` — bar chart from 50k run metrics
- `pytest_ci_output.txt` — 11 tests passed
- `api_health.txt`, `api_predict.txt` — local API smoke tests

**After Azure + GitHub (manual):**
- Azure Portal Web App + App Insights transaction search
- GitHub Actions green run screenshot → `github_actions_ci.png`
