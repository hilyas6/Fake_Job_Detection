from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from explain_ui import map_features_to_reasons
from model_runtime import load_model

st.set_page_config(page_title="Fake Job Detector", page_icon="🛡️", layout="centered")

st.title("🛡️ Detector")
st.caption("Likely fraud-risk screening using the deployed Improved TextGCN model.")

st.info(
    "**What this means:** **Fake** means fraud-like patterns were detected. "
    "**Real** means fewer fraud-like patterns were detected."
)

with st.expander("Disclaimer", expanded=True):
    st.write("Automated estimate; not a final verdict.")

EXAMPLES = {
    "Fake example": {
        "title": "Remote Data Entry Clerk - Immediate Hire",
        "description": "Earn up to $800 daily from home with no experience. Send your bank details and ID for onboarding today. "
        "Limited spots, urgent hiring, guaranteed income and no interview required. Contact us on Telegram now.",
    },
    "Real example": {
        "title": "Backend Software Engineer",
        "description": "Bright River Technologies Ltd is seeking a backend engineer with 3+ years of Python experience, "
        "REST API design, and PostgreSQL. Full-time London, UK role with benefits, structured interview process, "
        "clear responsibilities and requirements, and compensation of £65k per annum. Apply at https://brightriver.example/careers.",
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
    st.session_state.pop("last_explanation", None)

job_title = st.text_input("Job title", value=st.session_state.detector_input.get("title", ""))
job_description = st.text_area(
    "Job description",
    value=st.session_state.detector_input.get("description", ""),
    height=260,
    placeholder="Paste full job content, including requirements, compensation, and contact details.",
)

try:
    service = load_model()
    st.session_state.model_signature = service.model_signature
    st.session_state.threshold = service.threshold
except Exception as exc:
    st.error("Failed to load Improved TextGCN artifacts. If needed, run `git lfs pull`.")
    st.exception(exc)
    st.stop()

if st.button("Analyze", type="primary"):
    if not job_title.strip() or not job_description.strip():
        st.warning("Job title and description are required.")
        st.stop()

    text = f"{job_title}\n\n{job_description}"

    started = time.perf_counter()
    preprocessed = service.preprocess_text(text)
    prediction = service.predict_from_preprocessed(preprocessed)
    explanation = service.explain_text(text, mode="fast")
    ci_low, ci_high = service.estimate_uncertainty_interval(preprocessed, n=12)
    quality = service.input_quality(job_title, job_description)
    rel_bucket, rel_msg = service.reliability_bucket(ci_low, ci_high, quality)
    runtime_ms = (time.perf_counter() - started) * 1000.0

    st.session_state.last_prediction = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text": text,
        "title": job_title,
        "description": job_description,
        "text_length": quality["text_length"],
        "missing_fields": quality["missing_fields_list"],
        "label": prediction.label,
        "fake_probability": prediction.fake_probability,
        "real_probability": prediction.real_probability,
        "runtime_ms": runtime_ms,
        "threshold": service.threshold,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "reliability_bucket": rel_bucket,
        "reliability_msg": rel_msg,
        "model_signature": service.model_signature,
    }

    st.session_state.last_explanation = {
        "top_increase_fake": explanation.top_increase_fake,
        "top_decrease_fake": explanation.top_decrease_fake,
        "audit_top_increase_fake": explanation.audit_top_increase_fake,
        "audit_top_decrease_fake": explanation.audit_top_decrease_fake,
        "phrase_top_increase_fake": explanation.phrase_top_increase_fake,
        "phrase_top_decrease_fake": explanation.phrase_top_decrease_fake,
        "shap_error": explanation.shap_error,
        "mode": explanation.mode,
        "stability": explanation.stability,
    }

if "last_prediction" in st.session_state:
    p = st.session_state.last_prediction
    e = st.session_state.get("last_explanation", {})
    reasons = map_features_to_reasons(
        p["text"],
        e.get("phrase_top_increase_fake", []) + e.get("phrase_top_decrease_fake", []),
        e.get("top_increase_fake", []) + e.get("top_decrease_fake", []),
    )

    st.markdown("### Prediction summary")
    label_text = "LIKELY FAKE" if p["label"] == "fake" else "LIKELY REAL"
    color = "#d62728" if p["label"] == "fake" else "#2ca02c"
    st.markdown(f"<h2 style='color:{color}; margin-bottom:0.25rem;'>{label_text}</h2>", unsafe_allow_html=True)
    st.caption("Automated estimate; not a final verdict.")
    st.metric("Fake probability", f"{p['fake_probability']:.2f}")
    st.write(f"**Likely range (10–90%)**: {p['ci_low']:.2f} – {p['ci_high']:.2f}")
    st.write(f"**Reliability:** `{p['reliability_bucket']}` — {p['reliability_msg']}")
    if p["missing_fields"]:
        st.warning("Missing details detected: " + ", ".join(p["missing_fields"]))
    st.caption(f"Prediction generated in {p['runtime_ms']:.1f} ms")

    if st.button("Copy summary"):
        top3 = [r["title"] for r in reasons[:3]]
        summary = {
            "label": p["label"],
            "fake_prob": round(p["fake_probability"], 4),
            "range": [round(p["ci_low"], 4), round(p["ci_high"], 4)],
            "reliability": p["reliability_bucket"],
            "top_reasons": top3,
        }
        st.code(json.dumps(summary, ensure_ascii=False), language="json")

with st.sidebar:
    st.subheader("App pages")
    st.write("- Detector")
    st.write("- Explainability")
    st.success("Only deployed Improved TextGCN inference is enabled.")
    st.caption(f"Loaded model: {service.model_signature}")

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
