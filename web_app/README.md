# Web App (Streamlit)

This folder contains the new web application layer for the Fake Job Detection project.
It serves the improved TextGCN model and provides explainability outputs so users can understand *why* a listing is predicted as fake or real.

## What this app includes

- Improved TextGCN inference pipeline using:
  - `models/textgcn/textgcn_improved.pt`
  - `models/textgcn/graph_cache_improved.pt`
  - `models/textgcn/vectorizer_improved.joblib`
- Token-level masking-based TextGCN explanation (importance and protective tokens).
- SHAP-based explainability block to highlight decisive predictive traits.
- Streamlit interface for dynamic, user-facing classification.

The wording and presentation keep slight redundancies and mild passive style intentionally, as requested in the proposal constraints.

## Run locally

```bash
streamlit run web_app/app.py
```

If model files are tracked through Git LFS, run:

```bash
git lfs pull
```

before starting the app.
