from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from model_runtime import ImprovedTextGCNService

st.set_page_config(page_title="Explainable Fake Job Detector", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
        .shap-wrapper {
            background: #ffffff;
            border-radius: 0.75rem;
            padding: 0.65rem;
            border: 1px solid rgba(49, 51, 63, 0.15);
        }
        .shap-wrapper div {
            line-height: 1.35 !important;
            font-size: 13px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 Fake Job Classification + SHAP")
st.caption("TextGCN prediction with SHAP token-level explainability.")


@st.cache_resource(show_spinner=True)
def load_service() -> ImprovedTextGCNService:
    return ImprovedTextGCNService(
        model_dir=Path("models/textgcn"),
        metrics_path=Path("reports/metrics_textgcn_improved.csv"),
    )



try:
    import shap
except Exception:  # pragma: no cover - optional dependency handling
    shap = None


try:
    service = load_service()
except Exception as exc:
    st.error(
        "The improved TextGCN artifacts could not be loaded. "
        "If this repository is using Git LFS, run `git lfs pull` first."
    )
    st.exception(exc)
    st.stop()

job_text = st.text_area(
    "Paste a job listing text",
    height=240,
    placeholder="Include title, requirements, benefits, and contact details if available...",
)
run_btn = st.button("Classify and Explain", type="primary")

if run_btn:
    if not job_text.strip():
        st.warning("Please provide a non-empty job listing text before classification.")
        st.stop()

    result = service.explain_prediction(job_text)

    st.subheader("Classification")
    st.metric("Label", result.label.upper())

    st.subheader("SHAP Explainability")
    if shap is None:
        st.warning("SHAP is not installed in the environment.")
    elif result.shap_error:
        st.warning(result.shap_error)
    elif result.shap_values is None:
        st.warning("SHAP output was not available for this sample.")
    else:
        shap_values = result.shap_values
        try:
            shap_payload = shap_values[:, :, "fake_probability"][0]
        except Exception:
            shap_payload = shap_values[0]

        shap_height = 680
        shap_html_raw = shap.plots.text(shap_payload, display=False)
        shap_html = f"""
        <div class=\"shap-wrapper\">{shap_html_raw}</div>
        """
        components.html(shap_html, height=shap_height, scrolling=True)

st.markdown("---")
st.caption(
    "Built with Streamlit for practical fake-job detection and readable model explanations."
)
