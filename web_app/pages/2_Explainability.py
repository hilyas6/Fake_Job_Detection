from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from explain_ui import (
    build_counterfactual_checks,
    build_highlight_spans,
    bucket_magnitude,
    map_features_to_reasons,
    redact_emails,
    redact_phones,
    render_highlighted_html,
)
from model_runtime import load_model

st.set_page_config(page_title="Explainability", page_icon="🔍", layout="centered")
st.title("🔍 Explainability")
st.caption("Human-readable reasons and model signals for the latest analysis.")

service = load_model()
st.session_state.model_signature = service.model_signature
st.session_state.threshold = service.threshold

last_prediction = st.session_state.get("last_prediction")
last_explanation = st.session_state.get("last_explanation")
if not last_prediction or not last_explanation:
    st.info("Run an analysis on Detector first.")
    st.stop()

text = last_prediction["text"]
reason_signals = last_explanation.get("phrase_top_increase_fake", []) + last_explanation.get("phrase_top_decrease_fake", [])
token_signals = last_explanation.get("top_increase_fake", []) + last_explanation.get("top_decrease_fake", [])
reasons = map_features_to_reasons(text, reason_signals, token_signals)

if st.toggle("Recompute uncertainty with more MC samples (n=24)", value=False):
    preprocessed = service.preprocess_text(text)
    ci_low, ci_high = service.estimate_uncertainty_interval(preprocessed, n=24)
    quality = service.input_quality(last_prediction["title"], last_prediction["description"])
    bucket, msg = service.reliability_bucket(ci_low, ci_high, quality)
    last_prediction["ci_low"] = ci_low
    last_prediction["ci_high"] = ci_high
    last_prediction["reliability_bucket"] = bucket
    last_prediction["reliability_msg"] = msg

with st.container(border=True):
    st.markdown("### Summary")
    st.write(f"**Timestamp:** {last_prediction['timestamp']}")
    st.write(f"**Model:** {last_prediction.get('model_signature', service.model_signature)}")
    st.write(f"**Threshold:** {last_prediction.get('threshold', service.threshold):.2f}")
    st.write(f"**Label:** {last_prediction['label'].upper()}")
    st.write(
        f"**Fake probability:** {last_prediction['fake_probability']:.2f} "
        f"(range {last_prediction.get('ci_low', last_prediction['fake_probability']):.2f}–{last_prediction.get('ci_high', last_prediction['fake_probability']):.2f})"
    )
    st.write(f"**Reliability:** `{last_prediction.get('reliability_bucket', 'Unknown')}` — {last_prediction.get('reliability_msg', '')}")
    st.write("**Missing fields:** " + (", ".join(last_prediction.get("missing_fields", [])) or "None"))

    with st.expander("Limitations"):
        st.markdown("- May miss new scam patterns")
        st.markdown("- Not a legal determination")
        st.markdown("- Does not verify real-world company existence")
        st.markdown("- Non-English postings may be less reliable")

if st.button("Copy summary"):
    payload = {
        "label": last_prediction["label"],
        "fake_prob": round(last_prediction["fake_probability"], 4),
        "range": [round(last_prediction.get("ci_low", 0.0), 4), round(last_prediction.get("ci_high", 0.0), 4)],
        "reliability": last_prediction.get("reliability_bucket", "Unknown"),
        "top_reasons": [r["title"] for r in reasons[:3]],
    }
    st.code(json.dumps(payload, ensure_ascii=False), language="json")

reasons_tab, highlighted_tab, advanced_tab, feedback_tab = st.tabs(
    ["Reasons", "Highlighted text", "Model signals (advanced)", "Feedback"]
)

with reasons_tab:
    st.caption("These are correlations learned from training data; they are not proof.")
    if not reasons:
        st.warning("We can’t confidently explain this result. Please review the Advanced tab for raw signals.")
    else:
        for reason in reasons[:7]:
            direction_label = "Pushes Fake" if reason["direction"] == "pushes_fake" else "Pushes Real"
            with st.container(border=True):
                st.markdown(f"**{reason['title']}**")
                st.write(reason["explanation"])
                c1, c2 = st.columns([1, 1])
                c1.caption(direction_label)
                c2.caption(f"Strength: {reason['strength']}")
                if reason["evidence_snippets"]:
                    st.markdown("**Evidence snippets**")
                    for snippet in reason["evidence_snippets"]:
                        st.write(f"- {snippet}")

    st.markdown("### Safer next steps")
    for item in build_counterfactual_checks(reasons):
        st.write(f"- {item}")

with highlighted_tab:
    show_highlights = st.checkbox("Show highlights", value=True)
    show_raw = st.checkbox("Show raw text", value=False)
    if show_highlights:
        spans = build_highlight_spans(text, reasons)
        rendered = render_highlighted_html(text, spans)
        st.markdown(rendered, unsafe_allow_html=True)
    if show_raw:
        st.text_area("Raw text", value=text, height=250)

with advanced_tab:
    st.markdown("### Occlusion audit (most intuitive)")
    st.caption("If removing this phrase lowers fake risk, it was pushing the model toward FAKE.")
    audit = pd.DataFrame(last_explanation.get("audit_top_increase_fake", []) + last_explanation.get("audit_top_decrease_fake", []))
    if not audit.empty:
        buckets = bucket_magnitude(audit["impact"].tolist())
        audit["impact_bucket"] = buckets
        st.dataframe(audit[["feature", "impact", "impact_bucket", "tfidf_weight"]], use_container_width=True, hide_index=True)
        with st.expander("Raw occlusion values"):
            st.dataframe(audit, use_container_width=True, hide_index=True)
    else:
        st.write("No occlusion signals available.")

    st.markdown("### SHAP token signals")
    shap_error = last_explanation.get("shap_error")
    if shap_error:
        st.warning(shap_error)
    else:
        pos = pd.DataFrame(last_explanation.get("top_increase_fake", []))
        neg = pd.DataFrame(last_explanation.get("top_decrease_fake", []))
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Top tokens pushing toward Fake")
            st.dataframe(pos, use_container_width=True, hide_index=True)
        with c2:
            st.markdown("#### Top tokens pushing toward Real")
            st.dataframe(neg, use_container_width=True, hide_index=True)

    st.markdown("### Explanation quality")
    st.write(f"Mode: {last_explanation.get('mode', 'fast')}")
    if last_explanation.get("stability"):
        st.write(f"Stability: {last_explanation['stability']}")
    if shap_error:
        st.write(f"SHAP status: {shap_error}")

with feedback_tab:
    st.markdown("### Share feedback")
    choice = st.radio("How does this result look?", ["Seems correct", "Seems wrong", "Not sure"], horizontal=True)
    available_tags = [r["title"] for r in reasons[:5]] or ["No clear reasons"]
    selected_tags = st.multiselect("Reason categories", options=available_tags)
    comment = st.text_area("Tell us why (optional)", height=120)
    include_text = st.checkbox("Include anonymized text for research", value=False)

    if st.button("Submit feedback"):
        csv_path = APP_DIR / "feedback_log.csv"
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_signature": last_prediction.get("model_signature", service.model_signature),
            "threshold": last_prediction.get("threshold", service.threshold),
            "label": last_prediction["label"],
            "fake_prob": last_prediction["fake_probability"],
            "ci_low": last_prediction.get("ci_low", last_prediction["fake_probability"]),
            "ci_high": last_prediction.get("ci_high", last_prediction["fake_probability"]),
            "reliability_bucket": last_prediction.get("reliability_bucket", "Unknown"),
            "text_length": last_prediction.get("text_length", len(text)),
            "feedback_choice": choice,
            "selected_reason_tags": "|".join(selected_tags),
            "comment_length": len(comment.strip()),
            "anonymized_text": "",
        }
        if include_text:
            sanitized = redact_phones(redact_emails(text))
            row["anonymized_text"] = sanitized

        file_exists = csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        st.success("Thanks. Feedback saved.")
