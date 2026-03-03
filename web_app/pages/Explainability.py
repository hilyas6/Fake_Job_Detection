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
st.caption("Transparent reasons from the same Improved TextGCN prediction pipeline.")

last_prediction = st.session_state.get("last_prediction")
if not last_prediction:
    st.info("Run a prediction on the Detector page first.")
    st.stop()

st.markdown("### Last analyzed posting")
st.write(f"**Title:** {last_prediction['title']}")
st.caption(f"Label: **{last_prediction['label'].upper()}** · Fake probability: **{last_prediction['fake_probability']:.2f}**")

mode = st.toggle("Full Explanation", value=False, help="Fast Explanation (default) shows top-k reasons only.")
explain_mode = "full" if mode else "fast"

service = load_model()
explanation = service.explain_text(last_prediction["text"], mode=explain_mode)

if explanation.shap_error:
    st.warning(explanation.shap_error)
    st.stop()

st.markdown("### Top features that increase fake likelihood")
if explanation.top_increase_fake:
    df_pos = pd.DataFrame(explanation.top_increase_fake)
    st.bar_chart(df_pos.set_index("feature"))
    st.dataframe(df_pos, use_container_width=True, hide_index=True)
else:
    st.write("No positive contributors found for this sample.")

st.markdown("### Top features that decrease fake likelihood")
if explanation.top_decrease_fake:
    df_neg = pd.DataFrame(explanation.top_decrease_fake)
    df_neg["impact"] = df_neg["impact"].abs()
    st.bar_chart(df_neg.set_index("feature"))
    st.dataframe(df_neg, use_container_width=True, hide_index=True)
else:
    st.write("No negative contributors found for this sample.")

st.markdown("### Lightweight graph-aware summary")
st.write(
    "The Improved TextGCN uses word-node interactions in its graph encoder. "
    "The tokens above are the most influential nodes for this specific prediction."
)
