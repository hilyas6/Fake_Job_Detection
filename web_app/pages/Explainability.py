from __future__ import annotations

import pandas as pd
import streamlit as st

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from model_runtime import load_model

st.set_page_config(page_title="Explainability", page_icon="🔍", layout="centered")

st.title("🔍 Explainability")
st.caption("Primary SHAP explanation with an occlusion audit and phrase-level checks from the same Improved TextGCN pipeline.")

last_prediction = st.session_state.get("last_prediction")
if not last_prediction:
    st.info("Run a prediction on the Detector page first.")
    st.stop()

st.markdown("### Last analyzed posting")
st.write(f"**Title:** {last_prediction['title']}")
st.caption(f"Label: **{last_prediction['label'].upper()}** · Fake probability: **{last_prediction['fake_probability']:.2f}**")

mode = st.toggle("Full Explanation", value=False, help="Fast mode shows top reasons. Full mode adds a stability probe.")
explain_mode = "full" if mode else "fast"

service = load_model()
explanation = service.explain_text(last_prediction["text"], mode=explain_mode)

if explanation.shap_error:
    st.warning(explanation.shap_error)
    st.stop()

st.markdown("### Primary explanation (SHAP)")
st.write("Top token-level contributors based on SHAP attribution.")

st.markdown("#### Features that increase fake likelihood")
if explanation.top_increase_fake:
    df_pos = pd.DataFrame(explanation.top_increase_fake)
    st.bar_chart(df_pos.set_index("feature"))
    st.dataframe(df_pos, use_container_width=True, hide_index=True)
else:
    st.write("No positive contributors found for this sample.")

st.markdown("#### Features that decrease fake likelihood")
if explanation.top_decrease_fake:
    df_neg = pd.DataFrame(explanation.top_decrease_fake)
    df_neg["impact"] = df_neg["impact"].abs()
    st.bar_chart(df_neg.set_index("feature"))
    st.dataframe(df_neg, use_container_width=True, hide_index=True)
else:
    st.write("No negative contributors found for this sample.")

st.markdown("### Audit explanation (occlusion)")
st.write("Independent check: remove high-weight TF-IDF features and measure probability shifts.")

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Audit: increases fake likelihood")
    if explanation.audit_top_increase_fake:
        st.dataframe(pd.DataFrame(explanation.audit_top_increase_fake), use_container_width=True, hide_index=True)
    else:
        st.write("No positive audit contributors.")

with col2:
    st.markdown("#### Audit: decreases fake likelihood")
    if explanation.audit_top_decrease_fake:
        st.dataframe(pd.DataFrame(explanation.audit_top_decrease_fake), use_container_width=True, hide_index=True)
    else:
        st.write("No negative audit contributors.")

st.markdown("### Phrase-level attribution (n-gram aligned)")
phrase_pos = pd.DataFrame(explanation.phrase_top_increase_fake) if explanation.phrase_top_increase_fake else None
phrase_neg = pd.DataFrame(explanation.phrase_top_decrease_fake) if explanation.phrase_top_decrease_fake else None

if phrase_pos is None and phrase_neg is None:
    st.write("No phrase-level n-gram impacts found among top occlusion features for this sample.")
else:
    if phrase_pos is not None:
        st.markdown("#### Phrases increasing fake likelihood")
        st.dataframe(phrase_pos, use_container_width=True, hide_index=True)
    if phrase_neg is not None:
        st.markdown("#### Phrases decreasing fake likelihood")
        st.dataframe(phrase_neg, use_container_width=True, hide_index=True)

if explanation.stability:
    st.markdown("### Stability probe")
    st.metric("Top-feature rank stability (RBO@10)", f"{explanation.stability['rbo_top10']:.3f}")
    st.caption("RBO near 1.0 indicates stable top-ranked features under a light text perturbation.")

st.markdown("### Confidence and interpretation guardrails")
st.info(
    "Attributions indicate influential features for this prediction, not causality. "
    "Use SHAP and occlusion agreement as a confidence check, and treat low agreement as a review signal."
)
