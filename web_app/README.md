# Web App (Streamlit)

The Streamlit app is intentionally lightweight and contains exactly two pages:

- **Detector**: run fast fraud prediction with the deployed Improved TextGCN model.
- **Explainability**: show top reasons for the latest prediction using the same model pipeline.

## Deployed model artifacts

- `models/textgcn/textgcn_improved.pt`
- `models/textgcn/graph_cache_improved.pt`
- `models/textgcn/vectorizer_improved.joblib`

## Run

```bash
streamlit run web_app/app.py
```

If model files are tracked with Git LFS, run:

```bash
git lfs pull
```
