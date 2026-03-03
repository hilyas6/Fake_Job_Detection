from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from web_app.model_runtime import load_model

st.set_page_config(page_title="Fake Job Detector", page_icon="🛡️", layout="centered")

st.title("🛡️ Detector")
st.caption("Fast fraud-risk screening using the deployed Improved TextGCN model.")

st.info(
    "**What this means:** **Fake** means the model detected patterns consistent with fraudulent job ads. "
    "**Real** means fewer fraud-like patterns were detected."
)

with st.expander("Disclaimer", expanded=True):
    st.write("This is a research/demo tool. Predictions are probabilistic and should support, not replace, human review.")

EXAMPLES = {
    "Fake example": {
        "title": "Remote Data Entry Clerk - Immediate Hire",
        "description": "Earn up to $800 daily from home with no experience. Send your bank details and ID for onboarding today. "
        "Limited spots, urgent hiring, guaranteed income and no interview required.",
    },
    "Real example": {
        "title": "Backend Software Engineer",
        "description": "We are seeking a backend engineer with 3+ years of Python experience, REST API design, and PostgreSQL. "
        "Full-time role with benefits, structured interview process, and clear compensation range.",
    },
}

if "detector_input" not in st.session_state:
    st.session_state.detector_input = {"title": "", "description": ""}

c1, c2, c3 = st.columns([1, 1, 1])
example_choice = c1.selectbox("Example", options=list(EXAMPLES.keys()), label_visibility="collapsed")
if c2.button("Load example"):
    st.session_state.detector_input = EXAMPLES[example_choice].copy()
if c3.button("Clear"):
    st.session_state.detector_input = {"title": "", "description": ""}
    st.session_state.pop("last_prediction", None)

job_title = st.text_input("Job title", value=st.session_state.detector_input.get("title", ""))
job_description = st.text_area(
    "Job description",
    value=st.session_state.detector_input.get("description", ""),
    height=260,
    placeholder="Paste full job content, including requirements, compensation, and contact details.",
)

if st.button("Analyze", type="primary"):
    if not job_title.strip() or not job_description.strip():
        st.warning("Job title and description are required.")
        st.stop()

    text = f"{job_title}\n\n{job_description}"

    try:
        service = load_model()
    except Exception as exc:
        st.error("Failed to load Improved TextGCN artifacts. If needed, run `git lfs pull`.")
        st.exception(exc)
        st.stop()

    started = time.perf_counter()
    preprocessed = service.preprocess_text(text)
    prediction = service.predict_from_preprocessed(preprocessed)
    runtime_ms = (time.perf_counter() - started) * 1000.0

    st.session_state.last_prediction = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text": text,
        "title": job_title,
        "description": job_description,
        "text_length": len(text),
        "label": prediction.label,
        "fake_probability": prediction.fake_probability,
        "real_probability": prediction.real_probability,
        "runtime_ms": runtime_ms,
        "threshold": prediction.threshold,
    }

    st.markdown("### Prediction")
    label_text = "FAKE" if prediction.label == "fake" else "REAL"
    color = "#d62728" if prediction.label == "fake" else "#2ca02c"
    st.markdown(
        f"<h1 style='color:{color}; margin-bottom:0.25rem;'>{label_text}</h1>",
        unsafe_allow_html=True,
    )
    st.metric("Fake probability", f"{prediction.fake_probability:.2f}")
    st.caption("Key reasons will be shown on the Explainability page.")
    st.caption(f"Prediction generated in {runtime_ms:.1f} ms")

with st.sidebar:
    st.subheader("App pages")
    st.write("- Detector")
    st.write("- Explainability")
    st.success("Only deployed Improved TextGCN inference is enabled.")

log_path = Path("web_app/prediction_log.csv")
if "last_prediction" in st.session_state:
    p = st.session_state.last_prediction
    if st.button("Log latest prediction metadata"):
        line = f"{p['timestamp']},{p['text_length']},{p['label']},{p['fake_probability']:.4f},{p['runtime_ms']:.2f}\n"
        if not log_path.exists():
            log_path.write_text("timestamp,text_length,label,fake_probability,runtime_ms\n", encoding="utf-8")
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line)
        st.success("Metadata logged (no raw text stored).")
