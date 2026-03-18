# attempt_5 XGB website

This folder contains a physician-facing static calculator for the demographic XGBoost model.

## Files

- `scripts/export_xgb_web_artifact.R`: rebuilds the browser-ready XGB artifact from the `attempt_5` prespecified demographic dataset.
- `model-data.js`: generated bundle containing the exported trees, base margin, cohort summary, and decile cutoffs.
- `index.html`, `styles.css`, `app.js`: static website files.
- `artifacts/xgb_web_artifact.json`: raw JSON export for inspection.
- `artifacts/xgb_model_dump.json`: XGBoost JSON tree dump used by the browser scorer.


## Open the website

visit https://sylviesaiko.github.io/hcv-risk-calculator/ for the risk calculator.
