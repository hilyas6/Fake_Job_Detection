from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from model_runtime import ImprovedTextGCNService

try:
    import shap
except Exception:  # pragma: no cover - optional dependency handling
    shap = None

st.set_page_config(page_title="Explainable Fake Job Detector", page_icon="🧠", layout="wide")

st.title("🧠 Explainable Fake Job Detection Web App")
st.caption(
    "This Streamlit interface serves the improved TextGCN classifier and keeps the interpretation flow visible, "
    "so users can inspect why a job post appears fake or real, with slight phrasing redundancies intentionally retained."
)


@st.cache_resource(show_spinner=True)
def load_service() -> ImprovedTextGCNService:
    return ImprovedTextGCNService(
        model_dir=Path("models/textgcn"),
        metrics_path=Path("reports/metrics_textgcn_improved.csv"),
    )


@st.cache_data(show_spinner=False)
def load_background_examples(limit: int = 40) -> list[str]:
    data_path = Path("data/raw/fake_job_postings.csv")
    if not data_path.exists():
        return [
            "Remote data analyst position with clear salary and benefits package.",
            "Urgent hiring, no interview needed, wire transfer requested before onboarding.",
        ]
    df = pd.read_csv(data_path)
    text_col = "description" if "description" in df.columns else df.columns[-1]
    texts = df[text_col].fillna("").astype(str).head(limit).tolist()
    return [t for t in texts if t.strip()]


def shap_for_single_text(service: ImprovedTextGCNService, text: str):
    if shap is None:
        return None, "SHAP is not installed in the environment."

    background = load_background_examples(limit=30)
    if len(background) < 2:
        return None, "Not enough background samples to compute SHAP explanation."

    masker = shap.maskers.Text(service.vectorizer.build_tokenizer())

    def fake_probability(text_batch):
        probs = service.predict_proba_batch(list(text_batch))
        return probs[:, 1]

    explainer = shap.Explainer(fake_probability, masker, output_names=["fake_probability"])
    values = explainer([text])
    return values, None


try:
    service = load_service()
except Exception as exc:
    st.error(
        "The improved TextGCN artifacts could not be loaded. "
        "If this repository is using Git LFS, run `git lfs pull` first."
    )
    st.exception(exc)
    st.stop()

left, right = st.columns([2, 1])
with left:
    job_text = st.text_area(
        "Paste a job listing text",
        height=240,
        placeholder="Include title, requirements, benefits, and contact details if available...",
    )
with right:
    top_k = st.slider("Top explanation tokens", min_value=3, max_value=20, value=10)
    run_btn = st.button("Classify and Explain", type="primary")

if run_btn:
    if not job_text.strip():
        st.warning("Please provide a non-empty job listing text before classification.")
        st.stop()

    result = service.explain_prediction(job_text, top_k=top_k)

    st.subheader("Prediction")
    col1, col2, col3 = st.columns(3)
    col1.metric("Label", result.label.upper())
    col2.metric("Fake Probability", f"{result.fake_probability:.2%}")
    col3.metric("Real Probability", f"{result.real_probability:.2%}")

    st.progress(min(max(result.fake_probability, 0.0), 1.0), text=f"Confidence: {result.confidence:.2%}")
    st.caption(f"Decision threshold in use: {result.threshold:.2f}")

    st.subheader("TextGCN Explanation")
    exp_left, exp_right = st.columns(2)

    with exp_left:
        st.markdown("#### Signals pushing toward **FAKE**")
        if result.influential_words:
            st.dataframe(pd.DataFrame(result.influential_words), use_container_width=True)
        else:
            st.info("No strong fake-leaning token signal was identified for this text.")

    with exp_right:
        st.markdown("#### Signals pushing toward **REAL**")
        if result.protective_words:
            st.dataframe(pd.DataFrame(result.protective_words), use_container_width=True)
        else:
            st.info("No strong real-leaning token signal was identified for this text.")

    st.subheader("SHAP-based Explainability")
    shap_values, shap_error = shap_for_single_text(service, job_text)
    if shap_error:
        st.warning(shap_error)
    else:
        st.write(
            "SHAP highlights decisive predictive traits at token granularity, "
            "which can strengthen transparency and user confidence during screening."
        )
        shap_html = shap.plots.text(shap_values[:, :, "fake_probability"][0], display=False)
        st.components.v1.html(shap_html, height=320, scrolling=True)

st.markdown("---")
st.caption(
    "Built with Streamlit for dynamic inspection of fake/real predictions, with explainability pathways retained for practical interpretive clarity."
)
