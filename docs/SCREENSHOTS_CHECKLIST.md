# Screenshots checklist for soutenance and blog

All captures saved in `docs/screenshots/`:

## MLflow
- [x] `mlflow_model_comparison.png` — bar chart from 50k run metrics
- [x] `mlflow_ui_runs.png` — runs table (generated from model_comparison.json)

## API / Azure
- [x] `azure_api_health.png` — `/health` on Azure (model_loaded: true)
- [x] `azure_webapp.png` — Azure Web App endpoint
- [x] `api_health.txt`, `api_predict.txt`, `api_curl_local.txt` — local smoke tests

## Streamlit
- [x] `streamlit_demo.png` — prediction, feedback, alert after 3 errors

## Azure Application Insights
- [x] `app_insights_traces.png` — ALERT: 3 misclassified predictions in 5 minutes

## GitHub
- [x] `github_repo.png` — repository with code
- [x] `github_actions_ci.png` — green CI run #7
- [x] `pytest_ci_output.txt` — 11 tests passed

Regenerate chart/table artifacts:

```bash
python scripts/generate_screenshot_artifacts.py
```

Export presentation:

```bash
python scripts/export_presentation_pdf.py   # -> docs/presentation.pdf
pandoc docs/presentation.md -o docs/presentation.pptx --resource-path=docs
```
