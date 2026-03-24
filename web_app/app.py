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

from model_runtime import load_model

st.set_page_config(
    page_title="JobScan – Fake Job Detector",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Theme state ──────────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
dark_mode = st.session_state.dark_mode

# Handle theme toggle via query param (desired state encoded in URL)
_qp = st.query_params.get("set_theme", "")
if _qp == "light":
    st.session_state.dark_mode = False
    st.query_params.clear()
    st.rerun()
elif _qp == "dark":
    st.session_state.dark_mode = True
    st.query_params.clear()
    st.rerun()

def get_css(dark: bool) -> str:
    if dark:
        bg = """
            radial-gradient(ellipse 80% 65% at 0% 0%,   rgba(139,92,246,0.30) 0%, transparent 52%),
            radial-gradient(ellipse 65% 65% at 100% 100%,rgba(6,182,212,0.22) 0%, transparent 52%),
            radial-gradient(ellipse 55% 45% at 50% 50%, rgba(16,185,129,0.07) 0%, transparent 50%),
            radial-gradient(ellipse 40% 40% at 80% 20%, rgba(236,72,153,0.10) 0%, transparent 45%),
            linear-gradient(160deg,#07070F 0%,#0A0A16 50%,#060B14 100%)"""
        text_primary   = "#F1F5F9"
        text_secondary = "#94A3B8"
        text_muted     = "#64748B"
        card_bg_pb     = "rgba(10,10,22,0.38)"
        input_bg       = "rgba(255,255,255,0.05)"
        input_border   = "rgba(255,255,255,0.1)"
        input_color    = "#F1F5F9"
        input_ph       = "rgba(148,163,184,0.45)"
        label_color    = "#94A3B8"
        tab_list_bg    = "rgba(255,255,255,0.05)"
        tab_list_border= "rgba(255,255,255,0.08)"
        tab_color      = "rgba(148,163,184,0.8)"
        tab_hover_bg   = "rgba(139,92,246,0.12)"
        tab_hover_color= "#C4B5FD"
        tab_active_bg  = "rgba(139,92,246,0.2)"
        tab_active_col = "#DDD6FE"
        hdr_bg         = "rgba(7,7,15,0.84)"
        hdr_border     = "rgba(255,255,255,0.07)"
        hdr_shadow     = "0 1px 0 rgba(255,255,255,0.05),0 4px 32px rgba(0,0,0,0.5)"
        scroll_track   = "rgba(255,255,255,0.03)"
        sbtn_bg        = "rgba(255,255,255,0.05)"
        sbtn_color     = "#A78BFA"
        sbtn_border    = "rgba(167,139,250,0.28)"
        sbtn_hov_bg    = "rgba(139,92,246,0.14)"
        sbtn_hov_col   = "#C4B5FD"
        sbtn_hov_bdr   = "rgba(139,92,246,0.5)"
        theme_btn_bg   = "rgba(255,255,255,0.07)"
        theme_btn_col  = "#E2E8F0"
        theme_btn_bdr  = "rgba(255,255,255,0.15)"
        df_border      = "rgba(255,255,255,0.09)"
        checkbox_col   = "#94A3B8"
    else:
        bg = """
            radial-gradient(ellipse 80% 65% at 0% 0%,   rgba(139,92,246,0.12) 0%, transparent 52%),
            radial-gradient(ellipse 65% 65% at 100% 100%,rgba(6,182,212,0.10) 0%, transparent 52%),
            radial-gradient(ellipse 55% 45% at 50% 50%, rgba(16,185,129,0.05) 0%, transparent 50%),
            radial-gradient(ellipse 40% 40% at 80% 20%, rgba(236,72,153,0.06) 0%, transparent 45%),
            linear-gradient(160deg,#EEF2FF 0%,#F8FAFC 50%,#F0FDF4 100%)"""
        text_primary   = "#0F172A"
        text_secondary = "#374151"
        text_muted     = "#64748B"
        card_bg_pb     = "rgba(255,255,255,0.55)"
        input_bg       = "rgba(255,255,255,0.85)"
        input_border   = "rgba(139,92,246,0.2)"
        input_color    = "#0F172A"
        input_ph       = "rgba(100,116,139,0.5)"
        label_color    = "#374151"
        tab_list_bg    = "rgba(255,255,255,0.6)"
        tab_list_border= "rgba(139,92,246,0.15)"
        tab_color      = "#64748B"
        tab_hover_bg   = "rgba(139,92,246,0.08)"
        tab_hover_color= "#7C3AED"
        tab_active_bg  = "rgba(139,92,246,0.14)"
        tab_active_col = "#5B21B6"
        hdr_bg         = "rgba(255,255,255,0.80)"
        hdr_border     = "rgba(139,92,246,0.15)"
        hdr_shadow     = "0 1px 0 rgba(139,92,246,0.1),0 4px 24px rgba(0,0,0,0.06)"
        scroll_track   = "rgba(0,0,0,0.04)"
        sbtn_bg        = "rgba(255,255,255,0.8)"
        sbtn_color     = "#5B21B6"
        sbtn_border    = "rgba(139,92,246,0.3)"
        sbtn_hov_bg    = "rgba(139,92,246,0.08)"
        sbtn_hov_col   = "#4C1D95"
        sbtn_hov_bdr   = "rgba(139,92,246,0.55)"
        theme_btn_bg   = "rgba(139,92,246,0.1)"
        theme_btn_col  = "#4C1D95"
        theme_btn_bdr  = "rgba(139,92,246,0.3)"
        df_border      = "rgba(139,92,246,0.15)"
        checkbox_col   = "#374151"

    card_border_gradient = (
        "linear-gradient(135deg,rgba(139,92,246,0.65),rgba(6,182,212,0.55),"
        "rgba(236,72,153,0.45),rgba(16,185,129,0.45),rgba(139,92,246,0.65))"
    )

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Flash prevention ── */
[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],
[data-testid="stStatusWidget"],.stAppToolbar,#stDecoration,
header[data-testid="stHeader"] {{
    display:none !important;visibility:hidden !important;
    opacity:0 !important;height:0 !important;max-height:0 !important;overflow:hidden !important;
}}
[data-testid="stSidebar"],[data-testid="collapsedControl"]{{display:none !important;}}
#MainMenu,footer,.stDeployButton{{display:none !important;}}

/* ── Keyframes ── */
@keyframes fadeSlideUp {{
  from {{ opacity:0; transform:translateY(18px); }}
  to   {{ opacity:1; transform:translateY(0); }}
}}
@keyframes fadeIn {{
  from {{ opacity:0; }} to {{ opacity:1; }}
}}
@keyframes pulseDot {{
  0%,100% {{ box-shadow:0 0 0 0 rgba(52,211,153,0.7),0 0 8px rgba(52,211,153,0.4); }}
  50%      {{ box-shadow:0 0 0 8px rgba(52,211,153,0),0 0 16px rgba(52,211,153,0.2); }}
}}
@keyframes cardIn {{
  from {{ opacity:0; transform:translateY(14px) scale(0.99); }}
  to   {{ opacity:1; transform:translateY(0) scale(1); }}
}}

/* ── Background ── */
html, body, .stApp {{
    font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif !important;
    background: {bg} !important;
    background-attachment: fixed !important;
    color:{text_primary} !important;
}}

/* ── Layout ── */
.block-container {{
    padding-top:0 !important;
    padding-bottom:5rem !important;
    max-width:1160px !important;
    overflow:visible !important;
}}
section.main > div {{ overflow:visible !important; }}

/* ── Futuristic headings font ── */
h1,h2,h3 {{
    font-family:'Space Grotesk',sans-serif !important;
    letter-spacing:-0.02em !important;
}}

/* ── Primary button ── */
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg,#7C3AED,#4F46E5) !important;
    color:#FFFFFF !important;
    border:none !important;
    border-radius:10px !important;
    font-family:'Space Grotesk',sans-serif !important;
    font-weight:600 !important;
    font-size:15px !important;
    padding:0.65rem 1.6rem !important;
    letter-spacing:0.02em !important;
    box-shadow:0 0 14px rgba(124,58,237,0.45),0 2px 8px rgba(79,70,229,0.25),
               inset 0 1px 0 rgba(255,255,255,0.14) !important;
    transition:all 0.22s cubic-bezier(0.16,1,0.3,1) !important;
}}
.stButton > button[kind="primary"]:hover {{
    background: linear-gradient(135deg,#8B5CF6,#6D28D9) !important;
    box-shadow:0 0 22px rgba(139,92,246,0.52),0 4px 14px rgba(79,70,229,0.3),
               inset 0 1px 0 rgba(255,255,255,0.18) !important;
    transform:translateY(-2px) scale(1.01) !important;
}}
.stButton > button[kind="primary"]:active {{
    transform:translateY(0) scale(0.99) !important;
    box-shadow:0 0 10px rgba(124,58,237,0.35) !important;
    transition-duration:0.08s !important;
}}

/* ── Secondary button ── */
.stButton > button[kind="secondary"] {{
    background:{sbtn_bg} !important;
    color:{sbtn_color} !important;
    border:1px solid {sbtn_border} !important;
    border-radius:10px !important;
    font-family:'Space Grotesk',sans-serif !important;
    font-weight:500 !important;
    font-size:14px !important;
    box-shadow:none !important;
    transition:all 0.22s cubic-bezier(0.16,1,0.3,1) !important;
}}
.stButton > button[kind="secondary"]:hover {{
    background:{sbtn_hov_bg} !important;
    border-color:{sbtn_hov_bdr} !important;
    color:{sbtn_hov_col} !important;
    box-shadow:0 0 14px rgba(139,92,246,0.22) !important;
    transform:translateY(-1px) !important;
}}
.stButton > button[kind="secondary"]:active {{
    transform:translateY(0) scale(0.99) !important;
    transition-duration:0.08s !important;
}}

/* ── Page link ── */
[data-testid="stPageLink"] a {{
    display:inline-flex !important;
    align-items:center !important;
    justify-content:center !important;
    width:100% !important;
    gap:6px !important;
    background:linear-gradient(135deg,#7C3AED,#4F46E5) !important;
    color:#FFFFFF !important;
    padding:10px 20px !important;
    border-radius:10px !important;
    font-family:'Space Grotesk',sans-serif !important;
    font-weight:600 !important;
    font-size:14px !important;
    text-decoration:none !important;
    border:none !important;
    box-shadow:0 0 14px rgba(124,58,237,0.42) !important;
    transition:all 0.22s cubic-bezier(0.16,1,0.3,1) !important;
    letter-spacing:0.02em !important;
}}
[data-testid="stPageLink"] a:hover {{
    background:linear-gradient(135deg,#8B5CF6,#6D28D9) !important;
    box-shadow:0 0 22px rgba(139,92,246,0.52) !important;
    transform:translateY(-2px) !important;
    color:#FFFFFF !important;
}}
[data-testid="stPageLink"] a:active {{
    transform:translateY(0) !important;
    transition-duration:0.08s !important;
}}

/* ── Inputs (NOT liquid glass — kept clean) ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {{
    border-radius:10px !important;
    border:1px solid {input_border} !important;
    font-family:'Inter',sans-serif !important;
    font-size:14px !important;
    background:{input_bg} !important;
    color:{input_color} !important;
    transition:border-color 0.2s ease,box-shadow 0.2s ease !important;
}}
.stTextInput > div > div > input {{ padding:0.65rem 0.9rem !important; }}
.stTextArea > div > div > textarea {{ line-height:1.75 !important; resize:vertical !important; }}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {{
    border-color:rgba(139,92,246,0.7) !important;
    box-shadow:0 0 0 3px rgba(139,92,246,0.18),0 0 12px rgba(139,92,246,0.12) !important;
    outline:none !important;
}}
.stTextInput label,.stTextArea label {{
    font-family:'Space Grotesk',sans-serif !important;
    font-weight:600 !important;
    font-size:14px !important;
    color:{label_color} !important;
    letter-spacing:0.01em !important;
}}
.stTextInput > div > div > input::placeholder,
.stTextArea > div > div > textarea::placeholder {{ color:{input_ph} !important; }}

/* ── Liquid glass cards ── */
[data-testid="stVerticalBlockBorderWrapper"] {{
    border-radius:20px !important;
    background:
        linear-gradient({card_bg_pb},{card_bg_pb}) padding-box,
        {card_border_gradient} border-box !important;
    border:1px solid transparent !important;
    backdrop-filter:blur(28px) saturate(180%) !important;
    -webkit-backdrop-filter:blur(28px) saturate(180%) !important;
    box-shadow:
        0 8px 40px rgba(0,0,0,0.35),
        inset 0 1px 0 rgba(255,255,255,0.22),
        inset 0 -1px 0 rgba(0,0,0,0.08) !important;
    animation:cardIn 0.45s cubic-bezier(0.16,1,0.3,1) both !important;
    transition:box-shadow 0.3s ease !important;
}}
[data-testid="stVerticalBlockBorderWrapper"]:hover {{
    box-shadow:
        0 12px 50px rgba(0,0,0,0.42),
        0 0 28px rgba(139,92,246,0.12),
        inset 0 1px 0 rgba(255,255,255,0.28),
        inset 0 -1px 0 rgba(0,0,0,0.08) !important;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background:{tab_list_bg} !important;
    border-radius:12px !important;
    padding:4px !important;
    gap:2px !important;
    border:1px solid {tab_list_border} !important;
    backdrop-filter:blur(12px) !important;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius:9px !important;
    font-family:'Space Grotesk',sans-serif !important;
    font-weight:500 !important;
    font-size:14px !important;
    color:{tab_color} !important;
    padding:7px 20px !important;
    border:none !important;
    transition:all 0.2s cubic-bezier(0.16,1,0.3,1) !important;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color:{tab_hover_color} !important;
    background:{tab_hover_bg} !important;
}}
.stTabs [aria-selected="true"] {{
    background:{tab_active_bg} !important;
    color:{tab_active_col} !important;
    box-shadow:0 0 12px rgba(139,92,246,0.22),inset 0 1px 0 rgba(255,255,255,0.1) !important;
    font-weight:600 !important;
}}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {{ display:none !important; }}

/* ── Misc ── */
[data-testid="stSpinner"] > div {{ color:#A78BFA !important; }}
.stAlert {{ border-radius:12px !important; font-size:14px !important; }}
.stCheckbox label, .stToggle label {{ color:{checkbox_col} !important; font-size:14px !important; }}
.stRadio label {{ color:{checkbox_col} !important; font-size:14px !important; }}
[data-testid="stDataFrameContainer"] {{
    border-radius:12px !important;
    border:1px solid {df_border} !important;
    overflow:hidden !important;
    box-shadow:0 2px 16px rgba(0,0,0,0.2) !important;
}}
::-webkit-scrollbar {{ width:6px; height:6px; }}
::-webkit-scrollbar-track {{ background:{scroll_track}; border-radius:3px; }}
::-webkit-scrollbar-thumb {{ background:rgba(139,92,246,0.38); border-radius:3px; }}
::-webkit-scrollbar-thumb:hover {{ background:rgba(139,92,246,0.58); }}
</style>
"""

st.markdown(get_css(dark_mode), unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def risk_color(prob: float) -> str:
    if prob >= 0.65: return "#F87171"
    if prob >= 0.40: return "#FCD34D"
    return "#34D399"

def risk_bg(prob: float) -> str:
    if prob >= 0.65: return "rgba(239,68,68,0.12)"
    if prob >= 0.40: return "rgba(251,191,36,0.10)"
    return "rgba(52,211,153,0.10)"

def risk_border(prob: float) -> str:
    if prob >= 0.65: return "rgba(239,68,68,0.35)"
    if prob >= 0.40: return "rgba(251,191,36,0.35)"
    return "rgba(52,211,153,0.35)"


def gauge_html(prob: float) -> str:
    pct = round(prob * 100)
    color = risk_color(prob)
    marker = max(2, min(97, pct))
    return f"""
    <div style="margin:12px 0 8px;">
      <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px;">
        <span style="font-size:12px;font-weight:600;color:#64748B;
                     text-transform:uppercase;letter-spacing:0.07em;">Risk Score</span>
        <span style="font-size:30px;font-weight:800;color:{color};
                     letter-spacing:-0.03em;line-height:1;
                     text-shadow:0 0 16px {color}88;">{pct}%</span>
      </div>
      <div style="position:relative;height:12px;border-radius:6px;
                  background:linear-gradient(to right,#34D399 0%,#FCD34D 45%,#FB923C 75%,#F87171 100%);
                  box-shadow:0 0 10px rgba(0,0,0,0.3),inset 0 1px 3px rgba(0,0,0,0.3);">
        <div style="position:absolute;top:50%;left:{marker}%;
                    transform:translate(-50%,-50%);
                    width:20px;height:20px;border-radius:50%;
                    background:#1A1A2E;border:2.5px solid {color};
                    box-shadow:0 0 10px {color}99,0 2px 8px rgba(0,0,0,0.4);
                    transition:left 0.6s cubic-bezier(0.16,1,0.3,1);"></div>
      </div>
      <div style="display:flex;justify-content:space-between;margin-top:6px;">
        <span style="font-size:11px;color:#64748B;font-weight:500;">Low</span>
        <span style="font-size:11px;color:#64748B;font-weight:500;">Uncertain</span>
        <span style="font-size:11px;color:#64748B;font-weight:500;">High</span>
      </div>
    </div>
    """


def verdict_html(label: str, fake_prob: float, runtime_ms: float) -> str:
    is_fake  = label == "fake"
    accent   = "#F87171" if is_fake else "#34D399"
    bg       = risk_bg(fake_prob)
    border   = risk_border(fake_prob)
    verdict  = "Likely Fraudulent" if is_fake else "Likely Legitimate"
    sub      = "Fraud-like patterns detected in this posting." if is_fake else "No significant fraud patterns detected."
    dot      = "#DC2626" if is_fake else "#16A34A"
    return f"""
    <div style="background:{bg};border:1.5px solid {border};
                border-left:4px solid {accent};border-radius:14px;
                padding:18px 22px;margin-bottom:16px;
                box-shadow:0 0 20px {border},0 4px 24px rgba(0,0,0,0.3);
                animation:fadeSlideUp 0.4s cubic-bezier(0.16,1,0.3,1) both;">
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <div>
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:5px;">
            <div style="width:10px;height:10px;border-radius:50%;
                        background:{dot};flex-shrink:0;
                        animation:pulseDot 2.5s infinite;"></div>
            <span style="font-size:17px;font-weight:700;color:{accent};
                         letter-spacing:-0.01em;
                         text-shadow:0 0 12px {accent}55;">{verdict}</span>
          </div>
          <div style="font-size:13px;color:#94A3B8;font-weight:400;
                      padding-left:20px;line-height:1.5;">{sub}</div>
        </div>
        <div style="text-align:right;min-width:56px;padding-left:16px;flex-shrink:0;">
          <div style="font-size:11px;color:#64748B;font-weight:500;">analysed in</div>
          <div style="font-size:13px;font-weight:700;color:#CBD5E1;
                      margin-top:2px;">{runtime_ms:.0f} ms</div>
        </div>
      </div>
    </div>
    """


def signal_chip_html(sig: dict) -> str:
    is_fake  = sig["direction"] == "pushes_fake"
    accent   = "#F87171" if is_fake else "#34D399"
    tag_bg   = "rgba(239,68,68,0.10)"  if is_fake else "rgba(52,211,153,0.10)"
    tag_text = "Increases risk"        if is_fake else "Reduces risk"
    impact   = abs(sig.get("impact", 0.0))
    return f"""
    <div style="background:rgba(255,255,255,0.04);
                border:1px solid rgba(255,255,255,0.09);
                border-left:3px solid {accent};border-radius:12px;
                padding:10px 15px;margin-bottom:7px;
                backdrop-filter:blur(12px);
                box-shadow:0 2px 12px rgba(0,0,0,0.3),0 0 8px {accent}22;
                animation:fadeSlideUp 0.4s cubic-bezier(0.16,1,0.3,1) both;">
      <div style="display:flex;align-items:center;gap:8px;">
        <span style="font-size:13px;font-weight:600;color:#E2E8F0;flex:1;min-width:0;">
          {sig['feature']}
        </span>
        <span style="font-size:11px;color:#94A3B8;font-weight:500;">
          impact {impact:.3f}
        </span>
        <span style="background:{tag_bg};color:{accent};font-size:11px;
                     font-weight:600;padding:2px 9px;border-radius:99px;
                     white-space:nowrap;border:1px solid {accent}44;">{tag_text}</span>
      </div>
    </div>
    """


def empty_state_html() -> str:
    return """
    <div style="display:flex;flex-direction:column;align-items:center;
                justify-content:center;height:440px;text-align:center;padding:2rem;
                animation:fadeIn 0.6s ease both;">
      <div style="width:64px;height:64px;border-radius:16px;
                  background:rgba(139,92,246,0.12);
                  border:1px solid rgba(139,92,246,0.35);
                  display:flex;align-items:center;justify-content:center;
                  margin-bottom:20px;
                  box-shadow:0 0 28px rgba(139,92,246,0.2),inset 0 1px 0 rgba(255,255,255,0.06);">
        <div style="width:28px;height:28px;border-radius:50%;
                    border:2.5px solid rgba(139,92,246,0.6);"></div>
      </div>
      <div style="font-size:16px;font-weight:600;color:#64748B;margin-bottom:8px;">
        No analysis yet
      </div>
      <div style="font-size:13px;color:#475569;max-width:220px;line-height:1.7;font-weight:400;">
        Paste a job posting on the left and click Analyse to see results here.
      </div>
    </div>
    """


EXAMPLES = {
    "Suspicious": {
        "title": "Remote Data Entry Clerk – Immediate Hire",
        "description": (
            "Earn up to $800 daily from home with no experience needed. "
            "Send your bank details and a copy of your ID for onboarding today. "
            "Limited spots available — urgent hiring, guaranteed income, no interview required. "
            "Contact us on Telegram now to secure your position."
        ),
    },
    "Legitimate": {
        "title": "Backend Software Engineer",
        "description": (
            "Bright River Technologies Ltd is seeking a backend engineer with 3+ years of Python "
            "experience, REST API design, and PostgreSQL. Full-time role based in London, UK. "
            "We offer competitive benefits, a structured interview process, clear responsibilities "
            "and a compensation of £65,000 per annum. Apply at https://brightriver.example/careers."
        ),
    },
}

if "detector_input" not in st.session_state:
    st.session_state.detector_input = {"title": "", "description": ""}

try:
    service = load_model()
    st.session_state.model_signature = service.model_signature
    st.session_state.threshold = service.threshold
except Exception as exc:
    st.error("Could not load the TextGCN model. Run `python tuned_models/best_tuned_textgcn_model.py` first.")
    st.exception(exc)
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════════════════
_hdr_text   = "#F1F5F9" if dark_mode else "#0F172A"
_hdr_sub    = "#64748B"
_hdr_badge  = "rgba(255,255,255,0.07)" if dark_mode else "rgba(139,92,246,0.08)"
_hdr_bbdr   = "rgba(255,255,255,0.12)" if dark_mode else "rgba(139,92,246,0.2)"
_hdr_bcol   = "#94A3B8" if dark_mode else "#5B21B6"
_hdr_bg     = "rgba(7,7,15,0.84)" if dark_mode else "rgba(255,255,255,0.82)"
_hdr_border = "rgba(255,255,255,0.07)" if dark_mode else "rgba(139,92,246,0.15)"
_hdr_shadow = "0 1px 0 rgba(255,255,255,0.05),0 4px 32px rgba(0,0,0,0.5)" if dark_mode else "0 1px 0 rgba(139,92,246,0.1),0 4px 24px rgba(0,0,0,0.06)"
_glow       = "0 0 20px rgba(124,58,237,0.6),0 4px 12px rgba(79,70,229,0.4)"
_name_shadow= "0 0 20px rgba(139,92,246,0.4)" if dark_mode else "none"

st.markdown(f"""
<div style="background:{_hdr_bg};
            backdrop-filter:blur(22px);-webkit-backdrop-filter:blur(22px);
            border-bottom:1px solid {_hdr_border};
            padding:16px 36px;margin-bottom:32px;
            box-shadow:{_hdr_shadow};
            animation:fadeIn 0.4s ease both;
            position:sticky;top:0;z-index:999;">
  <div style="display:flex;align-items:center;gap:14px;max-width:1160px;margin:0 auto;">
    <div style="width:40px;height:40px;flex-shrink:0;
                background:linear-gradient(135deg,#7C3AED,#4F46E5);
                border-radius:12px;display:flex;align-items:center;justify-content:center;
                box-shadow:{_glow};">
      <div style="width:18px;height:18px;background:#FFFFFF;border-radius:4px;opacity:0.95;"></div>
    </div>
    <div style="flex:1;">
      <div style="font-size:20px;font-weight:700;color:{_hdr_text};
                  font-family:'Space Grotesk',sans-serif;
                  letter-spacing:-0.02em;line-height:1.2;
                  text-shadow:{_name_shadow};">JobScan</div>
      <div style="font-size:11px;color:{_hdr_sub};font-weight:400;
                  letter-spacing:0.02em;margin-top:1px;">Fake Job Detection</div>
    </div>
    <div style="display:flex;align-items:center;gap:8px;flex-shrink:0;">
      <div style="background:{_hdr_badge};border:1px solid {_hdr_bbdr};
                  border-radius:8px;padding:5px 13px;
                  font-size:12px;font-weight:500;color:{_hdr_bcol};
                  font-family:'Space Grotesk',sans-serif;
                  backdrop-filter:blur(8px);">
        TextGCN · Tuned
      </div>
      <a href="{'?set_theme=light' if dark_mode else '?set_theme=dark'}" target="_self"
         style="display:inline-flex;align-items:center;gap:6px;
                background:{'rgba(255,255,255,0.08)' if dark_mode else 'rgba(139,92,246,0.12)'};
                border:1px solid {'rgba(255,255,255,0.18)' if dark_mode else 'rgba(139,92,246,0.45)'};
                border-radius:8px;padding:5px 13px;
                font-size:12px;font-weight:600;
                color:{'#E2E8F0' if dark_mode else '#1E1B4B'};
                font-family:'Space Grotesk',sans-serif;
                backdrop-filter:blur(8px);
                text-decoration:none;cursor:pointer;
                transition:all 0.2s ease;">
        {'☀' if dark_mode else '◑'}&nbsp; {'Light' if dark_mode else 'Dark'}
      </a>
      <div style="display:flex;align-items:center;gap:7px;
                  background:rgba(52,211,153,0.10);border:1px solid rgba(52,211,153,0.35);
                  border-radius:8px;padding:5px 13px;
                  font-size:12px;font-weight:500;color:#34D399;
                  font-family:'Space Grotesk',sans-serif;
                  backdrop-filter:blur(8px);">
        <div style="width:7px;height:7px;border-radius:50%;background:#34D399;
                    animation:pulseDot 2s infinite;flex-shrink:0;"></div>
        Live
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Two-column layout
# ═══════════════════════════════════════════════════════════════════════════════
col_input, col_result = st.columns([5, 5], gap="large")

with col_input:
    with st.container(border=True):
        _card_title = "#E2E8F0" if dark_mode else "#0F172A"
        _card_sub   = "#64748B" if dark_mode else "#64748B"
        st.markdown(f"""
        <div style="padding:4px 4px 2px;">
          <div style="font-size:20px;font-weight:700;color:{_card_title};
                      font-family:'Space Grotesk',sans-serif;
                      letter-spacing:-0.02em;margin-bottom:5px;">Analyse a Job Posting</div>
          <div style="font-size:14px;color:{_card_sub};font-weight:400;margin-bottom:18px;
                      line-height:1.5;">
            Paste any job posting to screen it for fraud signals.
          </div>
        </div>
        """, unsafe_allow_html=True)

        job_title = st.text_input(
            "Job title",
            value=st.session_state.detector_input.get("title", ""),
            placeholder="e.g. Remote Data Entry Clerk",
        )
        job_description = st.text_area(
            "Job description",
            value=st.session_state.detector_input.get("description", ""),
            height=300,
            placeholder="Paste the full job posting here — include title, requirements, salary, and contact details.",
        )

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        c_btn, c_clear = st.columns([4, 1])
        with c_btn:
            analyze_clicked = st.button("Analyse Posting", type="primary", width="stretch")
        with c_clear:
            if st.button("Clear", width="stretch"):
                st.session_state.detector_input = {"title": "", "description": ""}
                st.session_state.pop("last_prediction", None)
                st.session_state.pop("last_explanation", None)
                st.rerun()

        st.markdown("""
        <div style="border-top:1px solid rgba(255,255,255,0.07);margin:18px -4px 12px;
                    padding-top:16px;padding-left:4px;padding-right:4px;">
          <div style="font-size:11px;font-weight:600;color:#475569;
                      text-transform:uppercase;letter-spacing:0.07em;margin-bottom:10px;">
            Try an example
          </div>
        </div>
        """, unsafe_allow_html=True)

        ex_c1, ex_c2 = st.columns(2)
        with ex_c1:
            if st.button("Suspicious posting", width="stretch"):
                st.session_state.detector_input = EXAMPLES["Suspicious"].copy()
                st.session_state.pop("last_prediction", None)
                st.session_state.pop("last_explanation", None)
                st.rerun()
        with ex_c2:
            if st.button("Legitimate posting", width="stretch"):
                st.session_state.detector_input = EXAMPLES["Legitimate"].copy()
                st.session_state.pop("last_prediction", None)
                st.session_state.pop("last_explanation", None)
                st.rerun()

        st.markdown("""
        <div style="margin-top:16px;padding:10px 14px;
                    background:rgba(255,255,255,0.03);
                    border-radius:10px;border:1px solid rgba(255,255,255,0.07);">
          <div style="font-size:11px;color:#475569;line-height:1.7;font-weight:400;">
            <span style="color:#64748B;font-weight:500;">Disclaimer:</span>
            Automated estimate based on language patterns — not a definitive verdict.
            Always verify job postings independently.
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── Run analysis ───────────────────────────────────────────────────────────────
if analyze_clicked:
    if not job_title.strip() or not job_description.strip():
        with col_input:
            st.warning("Please enter both a job title and description.")
    else:
        st.session_state.detector_input = {"title": job_title, "description": job_description}
        text = f"{job_title}\n\n{job_description}"
        with col_result:
            with st.spinner("Analysing…"):
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
            "text": text, "title": job_title, "description": job_description,
            "text_length": quality["text_length"],
            "missing_fields": quality["missing_fields_list"],
            "label": prediction.label,
            "fake_probability": prediction.fake_probability,
            "real_probability": prediction.real_probability,
            "runtime_ms": runtime_ms,
            "threshold": service.threshold,
            "ci_low": ci_low, "ci_high": ci_high,
            "reliability_bucket": rel_bucket, "reliability_msg": rel_msg,
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
            "mode": explanation.mode, "stability": explanation.stability,
        }
        st.rerun()

# ── Result panel ───────────────────────────────────────────────────────────────
with col_result:
    if "last_prediction" not in st.session_state:
        st.markdown(empty_state_html(), unsafe_allow_html=True)
    else:
        p = st.session_state.last_prediction
        e = st.session_state.get("last_explanation", {})
        signals = (
            [{"feature": r["feature"], "impact": r["impact"], "direction": "pushes_fake"}
             for r in e.get("top_increase_fake", [])[:3]] +
            [{"feature": r["feature"], "impact": r["impact"], "direction": "pushes_real"}
             for r in e.get("top_decrease_fake", [])[:2]]
        )

        with st.container(border=True):
            st.markdown("<div style='padding:4px 4px 0;'>", unsafe_allow_html=True)

            st.markdown(verdict_html(p["label"], p["fake_probability"], p["runtime_ms"]),
                        unsafe_allow_html=True)
            st.markdown(gauge_html(p["fake_probability"]), unsafe_allow_html=True)
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

            col_ci, col_rel = st.columns(2)
            with col_ci:
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.05);
                            border:1px solid rgba(255,255,255,0.09);
                            border-radius:12px;padding:14px 16px;
                            box-shadow:0 2px 16px rgba(0,0,0,0.35);">
                  <div style="font-size:11px;font-weight:600;color:#64748B;
                              text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">
                    Confidence range
                  </div>
                  <div style="font-size:18px;font-weight:700;color:#E2E8F0;
                              letter-spacing:-0.02em;">{p['ci_low']:.0%} – {p['ci_high']:.0%}</div>
                  <div style="font-size:11px;color:#475569;margin-top:3px;">10th – 90th percentile</div>
                </div>
                """, unsafe_allow_html=True)
            with col_rel:
                bucket = p.get("reliability_bucket", "Medium")
                b_styles = {
                    "High":   ("rgba(52,211,153,0.10)","rgba(52,211,153,0.3)","#34D399"),
                    "Medium": ("rgba(251,191,36,0.10)","rgba(251,191,36,0.3)","#FCD34D"),
                    "Low":    ("rgba(239,68,68,0.10)","rgba(239,68,68,0.3)","#F87171"),
                }
                b_bg, b_border, b_fg = b_styles.get(bucket, ("rgba(255,255,255,0.05)","rgba(255,255,255,0.09)","#94A3B8"))
                st.markdown(f"""
                <div style="background:{b_bg};border:1px solid {b_border};
                            border-radius:12px;padding:14px 16px;
                            box-shadow:0 2px 16px rgba(0,0,0,0.35),0 0 12px {b_fg}22;">
                  <div style="font-size:11px;font-weight:600;color:#64748B;
                              text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">
                    Reliability
                  </div>
                  <div style="font-size:18px;font-weight:700;color:{b_fg};
                              letter-spacing:-0.02em;text-shadow:0 0 10px {b_fg}66;">{bucket}</div>
                  <div style="font-size:11px;color:#475569;margin-top:3px;line-height:1.4;">
                    {p.get('reliability_msg','')}
                  </div>
                </div>
                """, unsafe_allow_html=True)

            if p.get("missing_fields"):
                st.markdown(f"""
                <div style="background:rgba(251,191,36,0.08);border:1px solid rgba(251,191,36,0.3);
                            border-radius:10px;padding:10px 14px;margin-top:12px;
                            font-size:12px;color:#FCD34D;line-height:1.5;">
                  <span style="font-weight:600;">Missing details:</span>
                  {', '.join(p['missing_fields'])} — adding these may improve accuracy.
                </div>
                """, unsafe_allow_html=True)

            if signals:
                _sig_title = "#E2E8F0" if dark_mode else "#0F172A"
                st.markdown(f"""
                <div style="margin-top:18px;margin-bottom:10px;">
                  <div style="font-size:16px;font-weight:700;color:{_sig_title};
                              font-family:'Space Grotesk',sans-serif;letter-spacing:-0.02em;">
                    Top signals detected
                  </div>
                  <div style="font-size:13px;color:#64748B;margin-top:2px;font-weight:400;">
                    Tokens that most influenced this result
                  </div>
                </div>
                """, unsafe_allow_html=True)
                for sig in signals:
                    st.markdown(signal_chip_html(sig), unsafe_allow_html=True)

            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            st.page_link("pages/2_Explainability.py", label="View full explanation  →", width="stretch")

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            if st.button("Copy summary as JSON", width="stretch"):
                summary = {
                    "label": p["label"],
                    "fake_probability": round(p["fake_probability"], 4),
                    "confidence_range": [round(p["ci_low"], 4), round(p["ci_high"], 4)],
                    "reliability": p.get("reliability_bucket", "Unknown"),
                    "top_signals": [s["feature"] for s in signals[:3]],
                    "timestamp": p["timestamp"],
                }
                st.code(json.dumps(summary, indent=2, ensure_ascii=False), language="json")

            st.markdown("</div>", unsafe_allow_html=True)

        log_path = APP_DIR / "prediction_log.csv"
        if st.button("Log this prediction (metadata only)", width="stretch"):
            line = f"{p['timestamp']},{p['text_length']},{p['label']},{p['fake_probability']:.4f},{p['runtime_ms']:.2f}\n"
            if not log_path.exists():
                log_path.write_text("timestamp,text_length,label,fake_probability,runtime_ms\n", encoding="utf-8")
            with log_path.open("a", encoding="utf-8") as f:
                f.write(line)
            st.success("Prediction metadata logged — no raw text stored.")
