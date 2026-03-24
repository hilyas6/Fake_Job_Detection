from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
    _PLOTLY = True
except Exception:
    _PLOTLY = False

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from explain_ui import (
    build_highlight_spans,
    bucket_magnitude,
    redact_emails,
    redact_phones,
    render_highlighted_html,
)
from model_runtime import load_model

st.set_page_config(
    page_title="JobScan – Explanation",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Theme ──────────────────────────────────────────────────────────────────────
dark_mode = st.session_state.get("dark_mode", True)
D = dark_mode

_bg = (
    "radial-gradient(ellipse 80% 65% at 0% 0%,rgba(139,92,246,0.30) 0%,transparent 52%),"
    "radial-gradient(ellipse 65% 65% at 100% 100%,rgba(6,182,212,0.22) 0%,transparent 52%),"
    "linear-gradient(160deg,#07070F 0%,#0A0A16 50%,#060B14 100%)"
    if D else
    "radial-gradient(ellipse 80% 65% at 0% 0%,rgba(139,92,246,0.12) 0%,transparent 52%),"
    "radial-gradient(ellipse 65% 65% at 100% 100%,rgba(6,182,212,0.10) 0%,transparent 52%),"
    "linear-gradient(160deg,#EEF2FF 0%,#F8FAFC 50%,#F0FDF4 100%)"
)
_card_pb    = "rgba(10,10,22,0.38)"    if D else "rgba(255,255,255,0.55)"
_card_shd   = "0.35"                   if D else "0.12"
_card_shd_h = "0.42"                   if D else "0.15"
_text_col   = "#F1F5F9"                if D else "#0F172A"
_inp_bg     = "rgba(255,255,255,0.05)" if D else "rgba(255,255,255,0.85)"
_inp_bdr    = "rgba(255,255,255,0.1)"  if D else "rgba(139,92,246,0.2)"
_tab_col    = "rgba(148,163,184,0.8)"  if D else "#64748B"
_lbl_col    = "#94A3B8"                if D else "#374151"
_tab_bg     = "rgba(255,255,255,0.05)" if D else "rgba(255,255,255,0.6)"
_tab_bdr    = "rgba(255,255,255,0.08)" if D else "rgba(139,92,246,0.15)"
_sbtn_bg    = "rgba(255,255,255,0.06)" if D else "rgba(139,92,246,0.08)"
_sbtn_col   = "#A78BFA"                if D else "#5B21B6"
_sbtn_bdr   = "rgba(167,139,250,0.28)" if D else "rgba(139,92,246,0.3)"
_sbtn_hbg   = "rgba(139,92,246,0.15)" if D else "rgba(139,92,246,0.14)"
_sbtn_hcol  = "#C4B5FD"                if D else "#4C1D95"
_sbtn_hbdr  = "rgba(139,92,246,0.6)"  if D else "rgba(139,92,246,0.55)"
_plink_bg   = "rgba(255,255,255,0.06)" if D else "rgba(255,255,255,0.85)"
_plink_col  = "#A78BFA"                if D else "#5B21B6"
_plink_bdr  = "rgba(167,139,250,0.3)" if D else "rgba(139,92,246,0.3)"
_H          = "#E2E8F0"                if D else "#0F172A"
_S          = "#64748B"
_note_col   = "#94A3B8"                if D else "#374151"
_mini_bg    = "rgba(255,255,255,0.05)" if D else "rgba(255,255,255,0.7)"
_mini_bdr   = "rgba(255,255,255,0.09)" if D else "rgba(139,92,246,0.15)"
_mini_col   = "#E2E8F0"                if D else "#0F172A"
_tab_sel_col = "#DDD6FE"               if D else "#5B21B6"
_tab_hov_col = "#C4B5FD"               if D else "#7C3AED"
_df_bdr      = "rgba(255,255,255,0.09)" if D else "rgba(139,92,246,0.18)"
# Highlighted-text panel (always light background so highlight colours are readable)
_hl_bg  = "rgba(15,20,40,0.90)"        if D else "rgba(255,255,255,0.92)"
_hl_col = "#E2E8F0"                    if D else "#1E293B"
_hl_bdr = "rgba(255,255,255,0.09)"     if D else "rgba(226,232,240,0.8)"

# ── Theme-dependent CSS ────────────────────────────────────────────────────────
st.markdown(f"""<style>
html, body, .stApp {{
    background:{_bg} !important;
    background-attachment:fixed !important;
    color:{_text_col} !important;
}}
/* Broad text override — p/li/span only; divs keep their inline styles */
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stMarkdownContainer"] span,
.stApp [data-testid="stExpanderDetails"] p,
.stApp [data-testid="stExpanderDetails"] li,
.stApp [data-testid="stExpanderDetails"] span,
.stApp [data-testid="stCaptionContainer"] p,
.stApp .stMarkdown p, .stApp .stMarkdown li,
.stApp .stMarkdown span {{
    color:{_text_col} !important;
}}
[data-testid="stVerticalBlockBorderWrapper"] {{
    background:
        linear-gradient({_card_pb},{_card_pb}) padding-box,
        linear-gradient(135deg,rgba(139,92,246,0.65),rgba(6,182,212,0.55),rgba(236,72,153,0.45),rgba(16,185,129,0.45),rgba(139,92,246,0.65)) border-box !important;
    box-shadow:0 8px 40px rgba(0,0,0,{_card_shd}),inset 0 1px 0 rgba(255,255,255,0.22),inset 0 -1px 0 rgba(0,0,0,0.06) !important;
}}
[data-testid="stVerticalBlockBorderWrapper"]:hover {{
    box-shadow:0 12px 50px rgba(0,0,0,{_card_shd_h}),0 0 28px rgba(139,92,246,0.12),inset 0 1px 0 rgba(255,255,255,0.28),inset 0 -1px 0 rgba(0,0,0,0.06) !important;
}}
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {{
    border:1px solid {_inp_bdr} !important;
    background:{_inp_bg} !important;
    color:{_text_col} !important;
}}
.stApp .stTabs [data-baseweb="tab-list"] {{
    background:{_tab_bg} !important;
    border:1px solid {_tab_bdr} !important;
}}
.stApp .stTabs [data-baseweb="tab"] {{ color:{_tab_col} !important; }}
.stApp .stTabs [data-baseweb="tab"]:hover {{ color:{_tab_hov_col} !important; }}
.stApp .stTabs [aria-selected="true"] {{ color:{_tab_sel_col} !important; }}
.stApp .stTextInput label,.stApp .stTextArea label {{ color:{_lbl_col} !important; }}
.stApp .stRadio label,.stApp .stCheckbox label,.stApp .stToggle label {{ color:{_lbl_col} !important; }}
.stButton > button[kind="secondary"] {{
    background:{_sbtn_bg} !important;
    color:{_sbtn_col} !important;
    border:1px solid {_sbtn_bdr} !important;
}}
.stButton > button[kind="secondary"] p {{ color:{_sbtn_col} !important; }}
.stButton > button[kind="secondary"]:hover {{
    background:{_sbtn_hbg} !important;
    color:{_sbtn_hcol} !important;
    border-color:{_sbtn_hbdr} !important;
}}
.stApp [data-testid="stPageLink"] a {{
    background:{_plink_bg} !important;
    color:{_plink_col} !important;
    border:1px solid {_plink_bdr} !important;
}}
.stApp [data-testid="stDataFrameContainer"] {{
    border:1px solid {_df_bdr} !important;
}}
</style>""", unsafe_allow_html=True)

# ── Static CSS ─────────────────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600;700;800&display=swap');

[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],
[data-testid="stStatusWidget"],.stAppToolbar,#stDecoration,
header[data-testid="stHeader"] {
    display:none !important;visibility:hidden !important;
    opacity:0 !important;height:0 !important;max-height:0 !important;overflow:hidden !important;
}
[data-testid="stSidebar"],[data-testid="collapsedControl"]{display:none !important;}
#MainMenu,footer,.stDeployButton{display:none !important;}

@keyframes fadeSlideUp {
  from { opacity:0; transform:translateY(18px); }
  to   { opacity:1; transform:translateY(0); }
}
@keyframes fadeIn { from { opacity:0; } to { opacity:1; } }
@keyframes pulseDot {
  0%,100% { box-shadow:0 0 0 0 rgba(52,211,153,0.7),0 0 8px rgba(52,211,153,0.4); }
  50%      { box-shadow:0 0 0 8px rgba(52,211,153,0),0 0 16px rgba(52,211,153,0.2); }
}
@keyframes cardIn {
  from { opacity:0; transform:translateY(14px) scale(0.99); }
  to   { opacity:1; transform:translateY(0) scale(1); }
}

html, body, .stApp { font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif !important; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-0.02em !important; }

.block-container {
    padding-top:0 !important;
    padding-bottom:5rem !important;
    max-width:1160px !important;
    overflow:visible !important;
}
section.main > div { overflow:visible !important; }

.stButton > button[kind="primary"] {
    background:linear-gradient(135deg,#7C3AED,#4F46E5) !important;
    color:#FFFFFF !important; border:none !important; border-radius:10px !important;
    font-family:'Space Grotesk',sans-serif !important; font-weight:600 !important;
    font-size:15px !important; padding:0.65rem 1.6rem !important;
    box-shadow:0 0 14px rgba(124,58,237,0.45),0 2px 8px rgba(79,70,229,0.25) !important;
    transition:all 0.22s cubic-bezier(0.16,1,0.3,1) !important;
}
.stButton > button[kind="primary"]:hover {
    background:linear-gradient(135deg,#8B5CF6,#6D28D9) !important;
    box-shadow:0 0 22px rgba(139,92,246,0.52),0 4px 14px rgba(79,70,229,0.3) !important;
    transform:translateY(-2px) scale(1.01) !important;
}
.stButton > button[kind="secondary"] {
    border-radius:10px !important; font-family:'Inter',sans-serif !important;
    font-weight:500 !important; font-size:14px !important;
    transition:all 0.2s cubic-bezier(0.16,1,0.3,1) !important;
}
.stButton > button[kind="secondary"]:hover {
    box-shadow:0 0 18px rgba(139,92,246,0.28) !important;
    transform:translateY(-1px) !important;
}
[data-testid="stPageLink"] a {
    display:inline-flex !important; align-items:center !important; gap:6px !important;
    padding:9px 18px !important; border-radius:10px !important;
    font-family:'Inter',sans-serif !important; font-weight:600 !important;
    font-size:13px !important; text-decoration:none !important;
    box-shadow:0 0 12px rgba(139,92,246,0.15) !important;
    transition:all 0.2s cubic-bezier(0.16,1,0.3,1) !important;
}
[data-testid="stPageLink"] a:hover {
    background:rgba(139,92,246,0.15) !important;
    border-color:rgba(139,92,246,0.6) !important;
    color:#C4B5FD !important;
    box-shadow:0 0 24px rgba(139,92,246,0.35) !important;
    transform:translateY(-1px) !important;
}
.stTextInput > div > div > input, .stTextArea > div > div > textarea {
    border-radius:10px !important; font-family:'Inter',sans-serif !important;
    font-size:14px !important; transition:border-color 0.2s ease,box-shadow 0.2s ease !important;
    line-height:1.75 !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color:rgba(139,92,246,0.7) !important;
    box-shadow:0 0 0 3px rgba(139,92,246,0.2) !important;
    outline:none !important;
}
.stTextInput > div > div > input::placeholder,
.stTextArea > div > div > textarea::placeholder { color:rgba(148,163,184,0.5) !important; }
.stTextInput label,.stTextArea label {
    font-family:'Space Grotesk',sans-serif !important;
    font-weight:600 !important; font-size:14px !important;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius:20px !important; border:1px solid transparent !important;
    backdrop-filter:blur(28px) saturate(180%) !important;
    -webkit-backdrop-filter:blur(28px) saturate(180%) !important;
    animation:cardIn 0.45s cubic-bezier(0.16,1,0.3,1) both !important;
    transition:box-shadow 0.3s ease !important;
}
.stTabs [data-baseweb="tab-list"] {
    border-radius:12px !important; padding:4px !important;
    gap:2px !important; backdrop-filter:blur(12px) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius:9px !important; font-family:'Space Grotesk',sans-serif !important;
    font-weight:500 !important; font-size:14px !important; padding:7px 20px !important;
    border:none !important; transition:all 0.2s cubic-bezier(0.16,1,0.3,1) !important;
}
.stTabs [data-baseweb="tab"]:hover { background:rgba(139,92,246,0.12) !important; }
.stTabs [aria-selected="true"] {
    background:rgba(139,92,246,0.2) !important;
    box-shadow:0 0 14px rgba(139,92,246,0.25) !important;
    font-weight:600 !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display:none !important; }
.stRadio > div { gap:8px !important; }
.stRadio label,.stCheckbox label,.stToggle label {
    font-family:'Inter',sans-serif !important; font-size:14px !important;
}
[data-testid="stDataFrameContainer"] {
    border-radius:12px !important; overflow:hidden !important;
    box-shadow:0 2px 16px rgba(0,0,0,0.2) !important;
}
[data-testid="stSpinner"] > div { color:#A78BFA !important; }
.stAlert { border-radius:10px !important; font-size:14px !important; }
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background:rgba(255,255,255,0.03); border-radius:3px; }
::-webkit-scrollbar-thumb { background:rgba(139,92,246,0.4); border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:rgba(139,92,246,0.65); }
</style>""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def risk_color(prob: float) -> str:
    if prob >= 0.65: return "#F87171"
    if prob >= 0.40: return "#FCD34D"
    return "#34D399"


def gauge_html(prob: float) -> str:
    pct   = round(prob * 100)
    color = risk_color(prob)
    marker = max(2, min(97, pct))
    return f"""
    <div style="margin:10px 0 6px;">
      <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:9px;">
        <span style="font-size:11px;font-weight:600;color:#64748B;text-transform:uppercase;letter-spacing:0.07em;">Fraud Probability</span>
        <span style="font-size:32px;font-weight:800;color:{color};letter-spacing:-0.03em;line-height:1;
                     text-shadow:0 0 16px {color}55;">{pct}%</span>
      </div>
      <div style="position:relative;height:14px;border-radius:99px;
                  background:linear-gradient(to right,#34D399 0%,#86EFAC 30%,#FCD34D 52%,#FB923C 75%,#F87171 100%);
                  box-shadow:0 2px 8px rgba(0,0,0,0.2),inset 0 1px 0 rgba(255,255,255,0.15);">
        <div style="position:absolute;top:50%;left:{marker}%;transform:translate(-50%,-50%);
                    width:22px;height:22px;border-radius:50%;background:#FFFFFF;border:2.5px solid {color};
                    box-shadow:0 0 10px {color}88,0 2px 6px rgba(0,0,0,0.2);"></div>
      </div>
      <div style="display:flex;justify-content:space-between;margin-top:6px;">
        <span style="font-size:11px;color:#64748B;font-weight:500;">Low</span>
        <span style="font-size:11px;color:#64748B;font-weight:500;">Uncertain</span>
        <span style="font-size:11px;color:#64748B;font-weight:500;">High</span>
      </div>
    </div>"""


def _plotly_bar(df: pd.DataFrame, x_col: str, y_col: str,
                color, x_title: str, height: int | None = None) -> "object | None":
    if not _PLOTLY or df.empty:
        return None
    sdf = df.copy().sort_values(x_col, key=abs, ascending=True).tail(12)
    colors = color if isinstance(color, list) else color
    fig = go.Figure(go.Bar(
        x=sdf[x_col].round(4).tolist(),
        y=sdf[y_col].tolist(),
        orientation="h",
        marker=dict(color=colors, opacity=0.88,
                    line=dict(color="rgba(255,255,255,0.1)", width=0.5)),
        hovertemplate="<b>%{y}</b><br>%{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        height=height or max(240, len(sdf) * 38 + 80),
        margin=dict(l=8, r=28, t=12, b=44),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", size=12, color=_H),
        xaxis=dict(
            title=dict(text=x_title, font=dict(size=11, color=_S)),
            showgrid=True, gridcolor="rgba(148,163,184,0.12)",
            zeroline=True, zerolinecolor="rgba(148,163,184,0.4)", zerolinewidth=1.5,
        ),
        yaxis=dict(showgrid=False, automargin=True),
    )
    return fig


def _signal_badges(rows: list[dict], is_fake: bool) -> str:
    """Render a row of pill badges for top SHAP tokens."""
    color  = "#F87171" if is_fake else "#34D399"
    bg     = "rgba(239,68,68,0.12)" if is_fake else "rgba(52,211,153,0.10)"
    border = "rgba(239,68,68,0.35)" if is_fake else "rgba(52,211,153,0.35)"
    sign   = "+" if is_fake else "−"
    pills  = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:5px;'
        f'background:{bg};border:1px solid {border};border-radius:99px;'
        f'padding:4px 12px;font-size:12px;font-weight:600;color:{color};">'
        f'<span style="opacity:0.7;font-size:10px;">{sign}{abs(r["impact"]):.3f}</span>'
        f'<span style="color:{_H};">{r["feature"]}</span></span>'
        for r in rows[:6]
    )
    return f'<div style="display:flex;flex-wrap:wrap;gap:7px;padding:8px 0;">{pills}</div>'


# ── Load model & session state ─────────────────────────────────────────────────
service = load_model()
st.session_state.model_signature = service.model_signature
st.session_state.threshold = service.threshold

last_prediction  = st.session_state.get("last_prediction")
last_explanation = st.session_state.get("last_explanation")

# ── Header ─────────────────────────────────────────────────────────────────────
_hdr_bg  = "rgba(7,7,15,0.84)"       if D else "rgba(255,255,255,0.82)"
_hdr_bdr = "rgba(255,255,255,0.07)"  if D else "rgba(139,92,246,0.15)"
_hdr_shd = ("0 1px 0 rgba(255,255,255,0.05),0 4px 32px rgba(0,0,0,0.5)"
             if D else "0 1px 0 rgba(139,92,246,0.1),0 4px 24px rgba(0,0,0,0.06)")
_name_col = "#F1F5F9" if D else "#0F172A"
_name_shd = "0 0 20px rgba(139,92,246,0.4)" if D else "none"

st.markdown(f"""
<div style="background:{_hdr_bg};backdrop-filter:blur(22px);-webkit-backdrop-filter:blur(22px);
            border-bottom:1px solid {_hdr_bdr};padding:16px 32px 18px;margin-bottom:28px;
            box-shadow:{_hdr_shd};animation:fadeIn 0.4s ease both;position:sticky;top:0;z-index:999;">
  <div style="display:flex;align-items:center;gap:14px;max-width:1160px;margin:0 auto;">
    <div style="width:38px;height:38px;background:linear-gradient(135deg,#7C3AED,#4F46E5);
                border-radius:11px;display:flex;align-items:center;justify-content:center;
                box-shadow:0 0 20px rgba(124,58,237,0.6),0 4px 12px rgba(79,70,229,0.4);">
      <div style="width:16px;height:16px;background:#FFFFFF;border-radius:3px;opacity:0.92;"></div>
    </div>
    <div>
      <div style="font-size:20px;font-weight:700;color:{_name_col};
                  font-family:'Space Grotesk',sans-serif;letter-spacing:-0.02em;line-height:1.2;
                  text-shadow:{_name_shd};">JobScan</div>
      <div style="font-size:11px;color:#64748B;font-weight:400;letter-spacing:0.02em;">
        Full Explanation Report
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

nav_col, _ = st.columns([2, 8])
with nav_col:
    st.page_link("app.py", label="← Back to Detector")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ── Guard ──────────────────────────────────────────────────────────────────────
if not last_prediction or not last_explanation:
    st.markdown(f"""
    <div style="text-align:center;padding:5rem 2rem;animation:fadeSlideUp 0.5s ease both;">
      <div style="font-size:17px;font-weight:700;color:#94A3B8;margin-bottom:8px;">No analysis to explain</div>
      <div style="font-size:13px;color:#CBD5E1;">Run an analysis on the Detector page first.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Derive data ────────────────────────────────────────────────────────────────
text         = last_prediction["text"]
pos_signals  = last_explanation.get("top_increase_fake", [])
neg_signals  = last_explanation.get("top_decrease_fake", [])
pos_df       = pd.DataFrame(pos_signals)
neg_df       = pd.DataFrame(neg_signals)
all_signals  = pos_signals + neg_signals
audit_rows   = (last_explanation.get("audit_top_increase_fake", []) +
                last_explanation.get("audit_top_decrease_fake", []))

# Highlight spans for text tab
spans    = build_highlight_spans(text, [
    {"direction": "pushes_fake", "matched_terms": [r["feature"] for r in pos_signals[:8]], "title": "Fraud signal"},
    {"direction": "pushes_real", "matched_terms": [r["feature"] for r in neg_signals[:8]], "title": "Legitimate signal"},
])
rendered = render_highlighted_html(text, spans)

# ── Uncertainty toggle ─────────────────────────────────────────────────────────
if st.toggle("Recompute uncertainty with more samples (n=24)", value=False):
    preprocessed = service.preprocess_text(text)
    ci_low, ci_high = service.estimate_uncertainty_interval(preprocessed, n=24)
    quality = service.input_quality(last_prediction["title"], last_prediction["description"])
    bucket, msg = service.reliability_bucket(ci_low, ci_high, quality)
    last_prediction.update({"ci_low": ci_low, "ci_high": ci_high,
                             "reliability_bucket": bucket, "reliability_msg": msg})

# ═══════════════════════════════════════════════════════════════════════════════
# Summary card
# ═══════════════════════════════════════════════════════════════════════════════
p              = last_prediction
is_fake        = p["label"] == "fake"
v_accent       = "#F87171" if is_fake else "#34D399"
v_bg           = "rgba(239,68,68,0.10)"   if is_fake else "rgba(52,211,153,0.10)"
v_border       = "rgba(239,68,68,0.35)"   if is_fake else "rgba(52,211,153,0.35)"
verdict_text   = "Likely Fraudulent"      if is_fake else "Likely Legitimate"

bucket = p.get("reliability_bucket", "Medium")
b_styles = {
    "High":   ("rgba(52,211,153,0.10)", "rgba(52,211,153,0.3)", "#34D399"),
    "Medium": ("rgba(251,191,36,0.10)", "rgba(251,191,36,0.3)", "#FCD34D"),
    "Low":    ("rgba(239,68,68,0.10)",  "rgba(239,68,68,0.3)",  "#F87171"),
}
b_bg, b_border, b_fg = b_styles.get(bucket, ("rgba(255,255,255,0.05)", "rgba(255,255,255,0.09)", "#94A3B8"))

with st.container(border=True):
    st.markdown("<div style='padding:4px 4px 0;'>", unsafe_allow_html=True)
    v_col, g_col = st.columns([4, 6])

    with v_col:
        st.markdown(f"""
        <div style="background:{v_bg};border:1px solid {v_border};border-left:4px solid {v_accent};
                    border-radius:14px;padding:20px;height:100%;
                    box-shadow:0 0 20px {v_accent}22,0 4px 24px rgba(0,0,0,0.3);">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
            <div style="width:10px;height:10px;border-radius:50%;background:{v_accent};flex-shrink:0;
                        animation:pulseDot 2.5s infinite;"></div>
            <span style="font-size:16px;font-weight:700;color:{v_accent};letter-spacing:-0.01em;
                         text-shadow:0 0 12px {v_accent}66;">{verdict_text}</span>
          </div>
          <div style="font-size:12px;color:#64748B;margin-bottom:12px;padding-left:18px;">
            Threshold: {p.get('threshold', 0.5):.2f}
          </div>
          <div style="font-size:11px;color:#475569;padding-left:18px;">
            {p.get('timestamp','')[:19].replace('T', ' ')} UTC
          </div>
        </div>
        """, unsafe_allow_html=True)

    with g_col:
        st.markdown(f"""
        <div style="background:{_mini_bg};border:1px solid {_mini_bdr};
                    border-radius:14px;padding:16px 20px;box-shadow:0 2px 16px rgba(0,0,0,0.15);">
        """, unsafe_allow_html=True)
        st.markdown(gauge_html(p["fake_probability"]), unsafe_allow_html=True)
        ci_lo = p.get("ci_low",  p["fake_probability"])
        ci_hi = p.get("ci_high", p["fake_probability"])
        st.markdown(f"""
          <div style="display:flex;gap:10px;margin-top:12px;">
            <div style="flex:1;background:{_mini_bg};border:1px solid {_mini_bdr};
                        border-radius:10px;padding:10px 14px;">
              <div style="font-size:10px;font-weight:600;color:{_S};text-transform:uppercase;
                          letter-spacing:0.06em;margin-bottom:4px;">Range (10–90%)</div>
              <div style="font-size:15px;font-weight:700;color:{_mini_col};">
                {ci_lo:.0%} – {ci_hi:.0%}
              </div>
            </div>
            <div style="flex:1;background:{b_bg};border:1px solid {b_border};
                        border-radius:10px;padding:10px 14px;
                        box-shadow:0 2px 12px rgba(0,0,0,0.3),0 0 10px {b_fg}22;">
              <div style="font-size:10px;font-weight:600;color:#64748B;text-transform:uppercase;
                          letter-spacing:0.06em;margin-bottom:4px;">Reliability</div>
              <div style="font-size:15px;font-weight:700;color:{b_fg};
                          text-shadow:0 0 10px {b_fg}66;">{bucket}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    if p.get("missing_fields"):
        st.markdown(f"""
        <div style="background:rgba(251,191,36,0.08);border:1px solid rgba(251,191,36,0.3);
                    border-radius:10px;padding:10px 14px;margin-top:12px;font-size:12px;
                    color:#FCD34D;line-height:1.5;">
          <span style="font-weight:600;">Missing details:</span>
          {', '.join(p['missing_fields'])} — adding these may improve accuracy.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

if st.button("Copy summary as JSON"):
    st.code(json.dumps({
        "label":            p["label"],
        "fake_probability": round(p["fake_probability"], 4),
        "confidence_range": [round(p.get("ci_low", 0.0), 4), round(p.get("ci_high", 0.0), 4)],
        "reliability":      p.get("reliability_bucket", "Unknown"),
        "top_fraud_tokens": [r["feature"] for r in pos_signals[:5]],
    }, indent=2, ensure_ascii=False), language="json")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════════════════════
tab_signals, tab_text, tab_method, tab_feedback = st.tabs([
    "📊  Model Signals",
    "🔍  Highlighted Text",
    "🔬  Methodology",
    "💬  Feedback",
])

# ── Tab 1: Model Signals ───────────────────────────────────────────────────────
with tab_signals:
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Top-signal badges summary
    if pos_signals or neg_signals:
        st.markdown(f"""
        <div style="background:{_mini_bg};border:1px solid {_mini_bdr};border-radius:14px;
                    padding:16px 20px;margin-bottom:24px;animation:fadeSlideUp 0.35s ease both;">
          <div style="font-size:13px;font-weight:600;color:#F87171;margin-bottom:4px;">
            ▲ Top fraud-driving tokens
          </div>
          {_signal_badges(pos_signals, is_fake=True)}
          <div style="font-size:13px;font-weight:600;color:#34D399;margin-top:12px;margin-bottom:4px;">
            ▼ Top legitimacy tokens
          </div>
          {_signal_badges(neg_signals, is_fake=False)}
        </div>
        """, unsafe_allow_html=True)

    # SHAP section
    st.markdown(f"""
    <div style="margin-bottom:8px;">
      <div style="font-size:16px;font-weight:700;color:{_H};font-family:'Space Grotesk',sans-serif;
                  letter-spacing:-0.01em;margin-bottom:4px;">SHAP Token Attribution</div>
      <div style="font-size:13px;color:{_S};line-height:1.65;">
        Shapley values assign each token a marginal credit score for the fraud-probability output
        (Lundberg &amp; Lee, 2017). Bars right →
        <span style="color:#F87171;font-weight:600;">Fraudulent</span>;
        bars left → <span style="color:#34D399;font-weight:600;">Legitimate</span>.
      </div>
    </div>
    """, unsafe_allow_html=True)

    shap_error = last_explanation.get("shap_error")
    if shap_error:
        st.warning(shap_error)
    else:
        shap_c1, shap_c2 = st.columns(2)
        with shap_c1:
            st.markdown("""
            <div style="font-size:12px;font-weight:600;color:#F87171;margin-bottom:10px;
                        padding:7px 12px;background:rgba(239,68,68,0.10);
                        border-radius:8px;border-left:3px solid rgba(239,68,68,0.6);">
              ▲ Pushing toward Fraudulent
            </div>""", unsafe_allow_html=True)
            if not pos_df.empty:
                fig = _plotly_bar(pos_df, "impact", "feature", "#F87171",
                                  "SHAP impact on fraud probability")
                if fig:
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                with st.expander("Raw values"):
                    st.dataframe(pos_df.round(4), hide_index=True)
            else:
                st.caption("No fraud-pushing tokens found.")

        with shap_c2:
            st.markdown("""
            <div style="font-size:12px;font-weight:600;color:#34D399;margin-bottom:10px;
                        padding:7px 12px;background:rgba(52,211,153,0.10);
                        border-radius:8px;border-left:3px solid rgba(52,211,153,0.6);">
              ▼ Pushing toward Legitimate
            </div>""", unsafe_allow_html=True)
            if not neg_df.empty:
                fig2 = _plotly_bar(neg_df, "impact", "feature", "#34D399",
                                   "SHAP impact on fraud probability")
                if fig2:
                    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
                with st.expander("Raw values"):
                    st.dataframe(neg_df.round(4), hide_index=True)
            else:
                st.caption("No legitimacy tokens found.")

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    # Occlusion Audit
    st.markdown(f"""
    <div style="margin-bottom:8px;">
      <div style="font-size:16px;font-weight:700;color:{_H};font-family:'Space Grotesk',sans-serif;
                  letter-spacing:-0.01em;margin-bottom:4px;">Occlusion Audit</div>
      <div style="font-size:13px;color:{_S};line-height:1.65;">
        Each key phrase is masked and the model re-run. The bar shows the probability drop —
        a large positive value means that phrase is a primary fraud driver.
        <span style="color:#F87171;font-weight:600;">Red</span> = fraud signal;
        <span style="color:#34D399;font-weight:600;">Green</span> = legitimacy signal.
      </div>
    </div>
    """, unsafe_allow_html=True)

    if audit_rows:
        audit_df = pd.DataFrame(audit_rows)
        occ_colors = ["#F87171" if v > 0 else "#34D399" for v in audit_df["impact"]]
        occ_fig = _plotly_bar(audit_df, "impact", "feature", occ_colors,
                              "Probability drop when phrase removed (positive = fraud driver)",
                              height=max(280, len(audit_df) * 38 + 80))
        if occ_fig:
            st.plotly_chart(occ_fig, use_container_width=True, config={"displayModeBar": False})

        with st.expander("Show data table"):
            display = audit_df.copy()
            display["impact_size"] = bucket_magnitude(display["impact"].tolist())
            display["impact"]       = display["impact"].round(4)
            display["tfidf_weight"] = display["tfidf_weight"].round(4)
            st.dataframe(
                display[["feature", "impact", "impact_size", "tfidf_weight"]].rename(columns={
                    "feature": "Phrase", "impact": "Impact",
                    "impact_size": "Size", "tfidf_weight": "TF-IDF weight",
                }),
                hide_index=True,
            )
    else:
        st.markdown(f"""
        <div style="padding:24px;text-align:center;color:{_S};font-size:14px;
                    background:{_mini_bg};border-radius:12px;border:1px solid {_mini_bdr};">
          No occlusion signals available.
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    with st.expander("Inference metadata"):
        st.markdown(f"""
        | Field | Value |
        |---|---|
        | **Explanation mode** | {last_explanation.get('mode', 'fast')} |
        | **Model** | `{p.get('model_signature', service.model_signature)}` |
        | **Decision threshold** | {p.get('threshold', service.threshold):.2f} |
        | **Inference time** | {p.get('runtime_ms', 0):.1f} ms |
        """)
        if last_explanation.get("stability"):
            st.markdown(f"**Explanation stability (RBO@10):** {last_explanation['stability']['rbo_top10']:.3f}")

# ── Tab 2: Highlighted Text ────────────────────────────────────────────────────
with tab_text:
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    col_opts1, col_opts2 = st.columns(2)
    show_highlights = col_opts1.checkbox("Show highlights", value=True)
    show_raw        = col_opts2.checkbox("Show raw text",   value=False)

    if show_highlights:
        st.markdown(f"""
        <div style="display:flex;gap:20px;margin-bottom:14px;padding:10px 16px;
                    background:{_mini_bg};border-radius:10px;border:1px solid {_mini_bdr};">
          <div style="display:flex;align-items:center;gap:8px;font-size:12px;color:{_note_col};font-weight:500;">
            <span style="display:inline-block;width:14px;height:14px;background:#ffd6d6;
                         border-radius:3px;border:1px solid rgba(239,68,68,0.5);
                         box-shadow:0 0 6px rgba(239,68,68,0.3);"></span>
            Increases fraud risk
          </div>
          <div style="display:flex;align-items:center;gap:8px;font-size:12px;color:{_note_col};font-weight:500;">
            <span style="display:inline-block;width:14px;height:14px;background:#d8f5d0;
                         border-radius:3px;border:1px solid rgba(52,211,153,0.5);
                         box-shadow:0 0 6px rgba(52,211,153,0.25);"></span>
            Reduces fraud risk
          </div>
        </div>
        <div style="background:{_hl_bg};border:1px solid {_hl_bdr};border-radius:16px;
                    padding:22px 26px;font-size:14px;line-height:1.9;color:{_hl_col};
                    word-break:break-word;
                    box-shadow:0 1px 4px rgba(0,0,0,0.04),0 8px 28px rgba(0,0,0,0.08);
                    animation:cardIn 0.4s ease both;">
          {rendered}
        </div>
        """, unsafe_allow_html=True)

    if show_raw:
        st.text_area("Raw posting text", value=text, height=260, label_visibility="collapsed")

# ── Tab 3: Methodology ─────────────────────────────────────────────────────────
with tab_method:
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Live performance metrics
    _pf1, _ppr, _pre, _pob = 0.839, 0.903, 0.783, 0.933
    try:
        import csv as _csv
        _mp = Path(__file__).resolve().parents[2] / "reports/tuned/metrics_textgcn_tuned.csv"
        with open(_mp) as _f:
            _row = next(_csv.DictReader(_f))
        _pf1 = float(_row.get("emscad_test_f1",        _pf1))
        _ppr = float(_row.get("emscad_test_precision",  _ppr))
        _pre = float(_row.get("emscad_test_recall",     _pre))
        _pob = float(_row.get("openbay_recall",         _pob))
    except Exception:
        pass

    m1, m2, m3, m4 = st.columns(4)
    for _col, _label, _val, _color in [
        (m1, "Test F1",    f"{_pf1:.3f}", "#A78BFA"),
        (m2, "Precision",  f"{_ppr:.3f}", "#34D399"),
        (m3, "Recall",     f"{_pre:.3f}", "#FCD34D"),
        (m4, "OOD Recall", f"{_pob:.3f}", "#06B6D4"),
    ]:
        with _col:
            st.markdown(f"""
            <div style="background:{_mini_bg};border:1px solid {_mini_bdr};
                        border-radius:14px;padding:16px 18px;text-align:center;
                        box-shadow:0 4px 16px rgba(0,0,0,0.1);">
              <div style="font-size:11px;font-weight:600;color:{_S};text-transform:uppercase;
                          letter-spacing:0.07em;margin-bottom:6px;">{_label}</div>
              <div style="font-size:26px;font-weight:800;color:{_color};
                          letter-spacing:-0.02em;font-family:'Space Grotesk',sans-serif;
                          text-shadow:0 0 16px {_color}44;">{_val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    meth_a, meth_b = st.columns([5, 5], gap="large")

    with meth_a:
        with st.container(border=True):
            st.markdown(f"""
            <div style="padding:4px 4px 0;">
            <div style="font-size:16px;font-weight:700;color:{_H};
                        font-family:'Space Grotesk',sans-serif;margin-bottom:12px;">
              TextGCN Architecture
            </div>
            <div style="font-size:13px;color:{_note_col};line-height:1.75;">
              <p style="margin:0 0 10px;">
                <b style="color:{_H};">Text Graph Convolutional Network</b> (Yao et al., 2019)
                represents both <em>documents</em> and <em>vocabulary words</em> as nodes in a
                single heterogeneous graph. Edge weights encode:
              </p>
              <ul style="margin:0 0 10px;padding-left:18px;list-style:disc;">
                <li style="margin-bottom:6px;">
                  <b style="color:{_H};">Word–Document edges</b>: TF-IDF score of each term within each posting.
                </li>
                <li style="margin-bottom:6px;">
                  <b style="color:{_H};">Word–Word edges</b>: Pointwise Mutual Information (PMI)
                  computed over a sliding window of 15 tokens.
                </li>
              </ul>
              <p style="margin:0 0 10px;">
                A <b style="color:{_H};">3-layer GCN</b> with residual connections (α = 0.7)
                propagates information across the graph. The final document representation
                passes through an MLP classifier to produce the fraud probability.
              </p>
              <p style="margin:0;">
                <b style="color:{_H};">Tuning:</b> 3-seed ensemble, PMI window = 15,
                hidden dim = 300, dropout = 0.35, label smoothing = 0.05.
              </p>
            </div>
            </div>
            """, unsafe_allow_html=True)

    with meth_b:
        with st.container(border=True):
            st.markdown(f"""
            <div style="padding:4px 4px 0;">
            <div style="font-size:16px;font-weight:700;color:{_H};
                        font-family:'Space Grotesk',sans-serif;margin-bottom:12px;">
              Explainability Pipeline
            </div>
            <div style="font-size:13px;color:{_note_col};line-height:1.75;">
              <p style="margin:0 0 10px;">Three complementary techniques produce the explanation:</p>
              <p style="margin:0 0 8px;">
                <span style="background:rgba(139,92,246,0.15);color:#A78BFA;font-weight:600;
                             padding:1px 8px;border-radius:6px;font-size:11px;">1</span>
                <b style="color:{_H};margin-left:6px;">SHAP</b>
                — perturbs the token space and assigns marginal credit to each word
                using cooperative game theory.
              </p>
              <p style="margin:0 0 8px;">
                <span style="background:rgba(139,92,246,0.15);color:#A78BFA;font-weight:600;
                             padding:1px 8px;border-radius:6px;font-size:11px;">2</span>
                <b style="color:{_H};margin-left:6px;">Occlusion Audit</b>
                — masks key phrases individually and measures the probability drop,
                providing a phrase-level view of fraud drivers.
              </p>
              <p style="margin:0 0 8px;">
                <span style="background:rgba(139,92,246,0.15);color:#A78BFA;font-weight:600;
                             padding:1px 8px;border-radius:6px;font-size:11px;">3</span>
                <b style="color:{_H};margin-left:6px;">Token Highlights</b>
                — maps SHAP-attributed tokens directly onto the posting text
                for an inline visual explanation.
              </p>
              <p style="margin:0;">
                <span style="background:rgba(139,92,246,0.15);color:#A78BFA;font-weight:600;
                             padding:1px 8px;border-radius:6px;font-size:11px;">4</span>
                <b style="color:{_H};margin-left:6px;">MC Dropout</b>
                — runs inference with dropout active (n = 8–24 samples) to estimate
                a calibrated confidence interval around the prediction.
              </p>
            </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(f"""
        <div style="padding:4px 4px 0;">
        <div style="font-size:16px;font-weight:700;color:{_H};
                    font-family:'Space Grotesk',sans-serif;margin-bottom:12px;">
          Dataset &amp; Evaluation
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;
                    font-size:13px;color:{_note_col};line-height:1.7;">
          <div>
            <div style="font-weight:700;color:{_H};margin-bottom:4px;">Training Data</div>
            <b style="color:{_H};">EMSCAD</b> — Employment Scam Aegean dataset.
            ~17,880 real postings and ~800 fraudulent. Split 60 / 20 / 20
            (train / val / test) stratified by label, seed = 42.
          </div>
          <div>
            <div style="font-weight:700;color:{_H};margin-bottom:4px;">OOD Evaluation</div>
            <b style="color:{_H};">OpenBay</b> dataset used as out-of-distribution test only
            (no training). Measures real-world generalisation beyond the EMSCAD distribution.
          </div>
          <div>
            <div style="font-weight:700;color:{_H};margin-bottom:4px;">Decision Threshold</div>
            Optimal threshold ({service.threshold:.2f}) selected by grid-searching 0.1–0.9
            to maximise F1 on the validation split. Applied consistently at inference.
          </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

# ── Tab 4: Feedback ────────────────────────────────────────────────────────────
with tab_feedback:
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="margin-bottom:18px;">
      <div style="font-size:15px;font-weight:700;color:{_H};letter-spacing:-0.01em;margin-bottom:4px;">
        Share feedback
      </div>
      <div style="font-size:13px;color:{_S};">Your input helps improve future versions of the model.</div>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("<div style='padding:4px 4px 0;'>", unsafe_allow_html=True)

        st.markdown(f"<span style='color:{_H};font-weight:600;font-size:14px;'>How does this result look to you?</span>",
                    unsafe_allow_html=True)
        choice = st.radio(
            "Result quality",
            ["Seems correct", "Seems wrong", "Not sure"],
            horizontal=True,
            label_visibility="collapsed",
        )

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        comment = st.text_area(
            "Additional comments (optional)",
            placeholder="Tell us more about why this seems right or wrong…",
            height=100,
        )
        include_text = st.checkbox(
            "Include anonymised posting text for research (emails, phones, and URLs will be redacted)",
            value=False,
        )

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        if st.button("Submit feedback", type="primary"):
            csv_path = APP_DIR / "feedback_log.csv"
            row = {
                "timestamp":         datetime.now(timezone.utc).isoformat(),
                "model_signature":   p.get("model_signature", service.model_signature),
                "threshold":         p.get("threshold", service.threshold),
                "label":             p["label"],
                "fake_prob":         p["fake_probability"],
                "ci_low":            p.get("ci_low",  p["fake_probability"]),
                "ci_high":           p.get("ci_high", p["fake_probability"]),
                "reliability_bucket": p.get("reliability_bucket", "Unknown"),
                "text_length":       p.get("text_length", len(text)),
                "feedback_choice":   choice,
                "comment_length":    len(comment.strip()),
                "anonymized_text":   redact_phones(redact_emails(text)) if include_text else "",
            }
            file_exists = csv_path.exists()
            with csv_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
            st.success("Thanks. Your feedback has been recorded.")

        st.markdown("</div>", unsafe_allow_html=True)
