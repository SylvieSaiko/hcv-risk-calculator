# XGB website

This folder contains a proof-of-concept static calculator for the demographic XGBoost model aligned to the paper's fixed-cycle primary analysis. The calculator is derived from NHANES data and is intended for research, education, and outreach-prioritization illustration. It has not been externally validated for standalone clinical decision-making.

## Files

- `scripts/export_xgb_web_artifact.R`: rebuilds the browser-ready XGB artifact from the `attempt_5` prespecified demographic dataset using the paper's fixed-cycle split, calibration, and validation-decile reference.
- `model-data.js`: generated bundle containing the exported trees, base margin, cohort summary, and decile cutoffs.
- `index.html`, `styles.css`, `app.js`: static website files.
- `artifacts/xgb_web_artifact.json`: raw JSON export for inspection.
- `artifacts/xgb_model_dump.json`: XGBoost JSON tree dump used by the browser scorer.


## Open the website

visit https://sylviesaiko.github.io/hcv-risk-calculator/ for the risk calculator.

## Version control and model access

The calculator code is version-controlled at https://github.com/SylvieSaiko/hcv-risk-calculator.

The browser-readable model object is available in `model-data.js`. The exported JSON artifact is available in `artifacts/xgb_web_artifact.json`, and the XGBoost tree dump used by the browser scorer is available in `artifacts/xgb_model_dump.json`.

## Privacy and data entry

Scoring is performed client-side in the browser after the page is opened. The calculator does not require names, record numbers, or other direct identifiers, and users should not enter direct identifiers into the page.
