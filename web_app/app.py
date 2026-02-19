from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from model_runtime import ImprovedTextGCNService

st.set_page_config(page_title="Explainable Fake Job Detector", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
        .shap-panel {
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 0.75rem;
            padding: 0.75rem;
            background: rgba(250, 250, 250, 0.5);
            margin-top: 0.25rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

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

        view_col, height_col = st.columns([3, 1])
        with view_col:
            full_width_view = st.checkbox(
                "Use full-width SHAP view",
                value=True,
                help="Turn this off if you prefer SHAP next to other content.",
            )
        with height_col:
            shap_height = st.slider("SHAP panel height", min_value=380, max_value=1200, value=760, step=20)

        st.write(
            "SHAP highlights decisive predictive traits at token granularity, "
            "which can strengthen transparency and user confidence during screening."
        )

        st.markdown(
            """
            <div class="shap-panel">
            Tip: Scroll inside the panel to inspect the full explanation, or increase panel height for long postings.
            </div>
            """,
            unsafe_allow_html=True,
        )

        shap_html = shap.plots.text(shap_payload, display=False)
        if full_width_view:
            placeholder = st.empty()
        else:
            left_col, right_col = st.columns([2, 1])
            with left_col:
                placeholder = st.empty()
            with right_col:
                st.info(
                    "Use the controls above to resize the SHAP section for complete token-level visibility."
                )

        with placeholder:
            st.components.v1.html(shap_html, height=shap_height, scrolling=True)

st.markdown("---")
st.caption(
    "Built with Streamlit for dynamic inspection of fake/real predictions, with explainability pathways retained for practical interpretive clarity."
)
