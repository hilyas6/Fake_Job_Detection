# Web App Guide

How to run and understand the Streamlit fraud detection app.

---

## Running the App

```bash
streamlit run web_app/app.py
```

Opens at `http://localhost:8501`.

If model files are tracked with Git LFS:
```bash
git lfs pull
```

---

## Design System

The app uses an **Apple iOS light design language** throughout both pages:

| Token | Value |
|---|---|
| Page background | `#f2f2f7` (iOS system grey) |
| Card background | `#ffffff` white, `border-radius: 16px` |
| Primary CTA | `#007aff` iOS blue |
| Primary text | `#1c1c1e` |
| Secondary text | `#8e8e93` |
| Tertiary / note text | `#3c3c43` |
| Font stack | `-apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Helvetica Neue', Arial` |

There is no dark mode and no theme toggle. The theme is applied via `.streamlit/config.toml` (`base = "light"`) combined with CSS injected in each page via `st.markdown(..., unsafe_allow_html=True)`.

---

## Pages

| Page | Description |
|---|---|
| **Detector** (`app.py`) | Paste a job posting → get a fraud probability with token-level highlights |
| **Explainability** (`pages/2_Explainability.py`) | Full report: SHAP attribution, occlusion audit, highlighted text, reliability score, model methodology |

The Explainability page reads results written to `st.session_state` by the Detector page. If no analysis has been run yet, it shows a placeholder and stops.

The Explainability page has five tabs:
- **Plain English** — fraud pattern categories matched from SHAP tokens + structural checklist
- **Model Signals** — SHAP bar charts and occlusion audit table (red = fraud driver, green = legitimacy driver)
- **Highlighted Text** — job posting text with fraud/legit words colour-coded inline
- **Methodology** — live performance metrics, TextGCN architecture summary, explainability pipeline
- **Feedback** — user feedback saved to `web_app/feedback_log.csv`

---

## Required Model Artifacts

The web app uses the **tuned TextGCN ensemble** from `models/tuned/textgcn_tuned/`:

```
models/tuned/textgcn_tuned/
├── textgcn_tuned.pt           # Model weights (3-seed ensemble averaged)
├── graph_cache_tuned.pt       # Pre-built PMI vocabulary graph
└── vectorizer_tuned.joblib    # TF-IDF vectoriser
```

If these are missing, generate them:
```bash
python tuned_models/best_tuned_textgcn_model.py
```

---

## Key Source Files

| File | Role |
|------|------|
| `web_app/app.py` | Detector page — input form, inference call, result card |
| `web_app/model_runtime.py` | `ImprovedWordGCN` model class, `ImprovedTextGCNService` inference + SHAP + MC Dropout |
| `web_app/explain_ui.py` | Pure-Python helpers: signal categorisation, highlight spans, plain-English summary, PII redaction |
| `web_app/pages/2_Explainability.py` | Full explanation report — five-tab layout |
| `.streamlit/config.toml` | Streamlit theme (`base = "light"`, iOS colour tokens) |

---

## Model Inference Details

**`ImprovedWordGCN`** (defined in `web_app/model_runtime.py`):
- 3-layer GCN with residual connections (α = 0.7), hidden dim = 300
- Graph nodes represent vocabulary terms; edges encode PMI (word–word) co-occurrence
- At inference the pre-built graph is loaded from `graph_cache_tuned.pt` — no graph construction at runtime
- Word embeddings are propagated once (`gcn()`) and cached; each new posting is processed via `forward_with_cached_word_h()`
- **MC Dropout** estimates uncertainty: runs N forward passes with dropout active, reports 10th–90th percentile range
- Predictions are bucketed into reliability levels (High / Medium / Low) based on MC Dropout variance and input quality

**Threshold:** loaded from `reports/tuned/metrics_textgcn_tuned.csv` at startup (default ~0.48).

---

## Tuned Models Summary

All tuned model artifacts are in `models/tuned/`:

| Model | Artifact | Script to regenerate |
|-------|----------|---------------------|
| LogReg, NB, RF, XGB, LGBM | `*.joblib` + `vectorizer.joblib` | `tuned_models/tune_classical_models.py` |
| Bi-LSTM | `bilstm.pt` | `tuned_models/tune_bilstm.py` |
| DistilBERT | `distilbert/` | `tuned_models/tune_distilbert.py` |
| TextGCN (production) | `textgcn_tuned/` | `tuned_models/best_tuned_textgcn_model.py` |

Tuning uses EMSCAD train/validation splits from `data/processed/splits.json`. The best validation-F1 configuration is selected, then evaluated on EMSCAD test + OpenBay.
