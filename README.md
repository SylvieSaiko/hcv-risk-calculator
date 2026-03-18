# attempt_5 XGB website

This folder contains a physician-facing static calculator for the `attempt_5` demographic XGBoost model.

## Files

- `scripts/export_xgb_web_artifact.R`: rebuilds the browser-ready XGB artifact from the `attempt_5` prespecified demographic dataset.
- `model-data.js`: generated bundle containing the exported trees, base margin, cohort summary, and decile cutoffs.
- `index.html`, `styles.css`, `app.js`: static website files.
- `artifacts/xgb_web_artifact.json`: raw JSON export for inspection.
- `artifacts/xgb_model_dump.json`: XGBoost JSON tree dump used by the browser scorer.

## Rebuild the model artifact

```bash
cd /Users/sylviesaiko/Desktop/guanwen/research_for_Dr_Chen
Rscript NHANES_MINE/attempt_5/website_xgb/scripts/export_xgb_web_artifact.R
```

## Open the website

You can open `index.html` directly in a browser because the model bundle is loaded as a local script.

If you prefer serving it locally:

```bash
cd /Users/sylviesaiko/Desktop/guanwen/research_for_Dr_Chen/NHANES_MINE/attempt_5/website_xgb
python3 -m http.server 8000
```

Then visit `http://127.0.0.1:8000`.
