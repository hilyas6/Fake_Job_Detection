from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from model_runtime import ImprovedTextGCNService

st.set_page_config(page_title="Explainable Fake Job Detector", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
        .shap-panel {
            border: 1px solid rgba(49, 51, 63, 0.15);
            border-radius: 0.75rem;
            padding: 0.75rem;
            background: rgba(250, 250, 250, 0.55);
            margin-top: 0.25rem;
        }
        .shap-summary-card {
            border: 1px solid rgba(49, 51, 63, 0.15);
            border-radius: 0.75rem;
            padding: 0.8rem 0.9rem;
            background: rgba(255, 255, 255, 0.7);
        }
        .shap-summary-card h4 {
            margin: 0 0 0.35rem 0;
            font-size: 0.98rem;
        }
        .shap-summary-card p {
            margin: 0;
            color: rgba(49, 51, 63, 0.9);
            font-size: 0.9rem;
        }
        .shap-wrapper {
            background: #ffffff;
            border-radius: 0.75rem;
            padding: 0.65rem;
            border: 1px solid rgba(49, 51, 63, 0.15);
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.2);
        }
        .shap-wrapper div {
            line-height: 1.35 !important;
            font-size: 13px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 Explainable Fake Job Detection Web App")
st.caption(
    "Classify a job post and review a clear explanation of why the model predicts FAKE or REAL."
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

        view_col, height_col = st.columns([3, 1])
        with view_col:
            full_width_view = st.checkbox(
                "Use full-width SHAP view",
                value=True,
                help="Turn this off if you prefer SHAP next to other content.",
            )
        with height_col:
            shap_height = st.slider("SHAP panel height", min_value=380, max_value=1200, value=760, step=20)

        st.markdown(
            """
            <div class="shap-panel">
                <b>How to read this view</b><br/>
                • Red tokens push the prediction toward <b>FAKE</b>.<br/>
                • Blue tokens push the prediction toward <b>REAL</b>.<br/>
                • Stronger color means a larger impact.
            </div>
            """,
            unsafe_allow_html=True,
        )

        shap_html_raw = shap.plots.text(shap_payload, display=False)
        shap_html = f"""
        <div class=\"shap-wrapper\">{shap_html_raw}</div>
        """
        if full_width_view:
            placeholder = st.empty()
        else:
            left_col, right_col = st.columns([2, 1])
            with left_col:
                placeholder = st.empty()
            with right_col:
                st.info(
                    "Use the controls above to adjust the SHAP panel size for better readability."
                )

        with placeholder:
            components.html(shap_html, height=shap_height, scrolling=True)

        st.markdown("#### Key token summary")
        summary_left, summary_right = st.columns(2)
        with summary_left:
            if result.influential_words:
                top_fake = result.influential_words[0]
                st.markdown(
                    f"""
                    <div class="shap-summary-card">
                        <h4>Top FAKE-driving token: <code>{top_fake['word']}</code></h4>
                        <p>
                            Impact on fake probability: <b>+{top_fake['impact_on_fake_probability']:.4f}</b><br/>
                            Frequency in text: <b>{int(top_fake['occurrences'])}</b>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info("No strong FAKE-driving tokens were detected.")

        with summary_right:
            if result.protective_words:
                top_real = result.protective_words[0]
                st.markdown(
                    f"""
                    <div class="shap-summary-card">
                        <h4>Top REAL-driving token: <code>{top_real['word']}</code></h4>
                        <p>
                            Impact on fake probability: <b>{top_real['impact_on_fake_probability']:.4f}</b><br/>
                            Frequency in text: <b>{int(top_real['occurrences'])}</b>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info("No strong REAL-driving tokens were detected.")

        detail_frames = []
        if result.influential_words:
            fake_df = pd.DataFrame(result.influential_words).copy()
            fake_df.insert(0, "signal", "FAKE")
            detail_frames.append(fake_df)
        if result.protective_words:
            real_df = pd.DataFrame(result.protective_words).copy()
            real_df.insert(0, "signal", "REAL")
            detail_frames.append(real_df)

        if detail_frames:
            detail_df = pd.concat(detail_frames, ignore_index=True)
            detail_df = detail_df[["signal", "word", "impact_on_fake_probability", "absolute_impact", "impact_strength", "occurrences"]]
            detail_df = detail_df.rename(
                columns={
                    "signal": "Signal",
                    "word": "Token",
                    "impact_on_fake_probability": "Impact on Fake Probability",
                    "absolute_impact": "Absolute Impact",
                    "impact_strength": "Impact Level",
                    "occurrences": "Occurrences",
                }
            )
            st.dataframe(
                detail_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Impact on Fake Probability": st.column_config.NumberColumn(format="%.4f"),
                    "Absolute Impact": st.column_config.NumberColumn(format="%.4f"),
                    "Occurrences": st.column_config.NumberColumn(format="%d"),
                },
            )
        else:
            st.info("No token-level SHAP details were available for interpretation.")

st.markdown("---")
st.caption(
    "Built with Streamlit for practical fake-job detection and readable model explanations."
)
