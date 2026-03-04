# Web App (Streamlit)

The Streamlit app is intentionally lightweight and contains exactly two pages:

- **Detector**: run fast fraud prediction with the deployed Improved TextGCN model.
- **Explainability**: show top reasons for the latest prediction using the same model pipeline.

## Explainability design

The Explainability page now exposes three aligned views from the same Improved TextGCN artifacts:

- **Primary SHAP token attribution** (default view).
- **Occlusion audit** over high-weight TF-IDF features to cross-check SHAP directionality.
- **Phrase-level attribution** extracted from n-gram features in the occlusion audit.

In **Full Explanation** mode, the app also computes a lightweight stability probe
(rank-biased overlap over top contributors after a small text perturbation).

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
