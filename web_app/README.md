# Web App (Streamlit)

This folder contains the new web application layer for the Fake Job Detection project.
It serves the improved TextGCN model and focuses the UI on only two outputs: the job classification label and SHAP token-level explainability.

## What this app includes

- Improved TextGCN inference pipeline using:
  - `models/textgcn/textgcn_improved.pt`
  - `models/textgcn/graph_cache_improved.pt`
  - `models/textgcn/vectorizer_improved.joblib`
- TextGCN-based prediction using the improved model artifacts.
- SHAP-based token explainability for the same TextGCN prediction path.
- Minimal Streamlit interface that removes extra result panels.
- Runtime optimizations in inference to reduce response latency.

## Run locally

```bash
streamlit run web_app/app.py
```

If model files are tracked through Git LFS, run:

```bash
git lfs pull
```

before starting the app.
