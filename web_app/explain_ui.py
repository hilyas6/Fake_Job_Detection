# explain_ui.py — UI helper functions for explainability (no Streamlit imports)
# Used by both app.py and pages/2_Explainability.py
from __future__ import annotations

import html
import re
from typing import Any


def _quantile_thresholds(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    sorted_vals = sorted(values)

    def pick(q: float) -> float:
        if len(sorted_vals) == 1:
            return sorted_vals[0]
        idx = max(0, min(int(round((len(sorted_vals) - 1) * q)), len(sorted_vals) - 1))
        return sorted_vals[idx]

    return pick(0.33), pick(0.66)


def bucket_strength(value: float, thresholds: tuple[float, float]) -> str:
    lo, hi = thresholds
    abs_v = abs(value)
    if abs_v >= hi:
        return "High"
    if abs_v >= lo:
        return "Medium"
    return "Low"


def bucket_magnitude(values: list[float]) -> list[str]:
    if not values:
        return []
    q1, q2 = _quantile_thresholds([abs(v) for v in values])
    return [
        bucket_strength(v, (q1, q2))
        .replace("High", "Large")
        .replace("Low", "Small")
        for v in values
    ]


def find_spans(text: str, term: str, max_hits: int = 3) -> list[tuple[int, int]]:
    if not text or not term:
        return []
    term = term.strip()
    if not term:
        return []
    is_alnum = bool(re.fullmatch(r"[\w\s]+", term))
    pattern = rf"\b{re.escape(term)}\b" if is_alnum else re.escape(term)
    spans = []
    for m in re.finditer(pattern, text, flags=re.IGNORECASE):
        spans.append((m.start(), m.end()))
        if len(spans) >= max_hits:
            break
    return spans


def build_highlight_spans(text: str, reasons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for reason in reasons:
        kind = "fake" if reason.get("direction") == "pushes_fake" else "real"
        for term in reason.get("matched_terms", []):
            for start, end in find_spans(text, term):
                spans.append({"start": start, "end": end, "kind": kind,
                               "label": reason.get("title", "signal")})

    spans.sort(key=lambda s: (s["start"], -(s["end"] - s["start"])))
    merged: list[dict[str, Any]] = []
    for span in spans:
        if not merged:
            merged.append(span)
            continue
        prev = merged[-1]
        if span["start"] < prev["end"]:
            if prev["kind"] == "fake":
                continue
            if span["kind"] == "fake":
                merged[-1] = span
            else:
                prev["end"] = max(prev["end"], span["end"])
        else:
            merged.append(span)
    return merged


def render_highlighted_html(text: str, spans: list[dict[str, Any]]) -> str:
    if not spans:
        return f"<div style='white-space:pre-wrap'>{html.escape(text)}</div>"

    parts = []
    cursor = 0
    for span in spans:
        start, end = int(span["start"]), int(span["end"])
        if start > cursor:
            parts.append(html.escape(text[cursor:start]))
        frag  = html.escape(text[start:end])
        color = "#ffd6d6" if span["kind"] == "fake" else "#d8f5d0"
        label = html.escape(str(span.get("label", "signal")))
        parts.append(
            f"<span title='{label}' style='background:{color};padding:0 2px;border-radius:3px;'>"
            f"{frag}</span>"
        )
        cursor = end
    if cursor < len(text):
        parts.append(html.escape(text[cursor:]))
    return "<div style='white-space:pre-wrap;line-height:1.6'>" + "".join(parts) + "</div>"


def redact_emails(text: str) -> str:
    return re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
                  "[REDACTED_EMAIL]", text)


def redact_phones(text: str) -> str:
    return re.sub(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)", "[REDACTED_PHONE]", text)


# Plain-English explainability

# Maps keyword categories to icons and template explanations shown to users
FRAUD_CATEGORIES: dict[str, dict] = {
    "Urgency & Pressure Language": {
        "keywords": {
            "urgent", "urgently", "immediately", "asap", "instant", "quick",
            "fast", "hurry", "today", "now", "deadline", "limited", "rush",
            "speedy", "prompt", "quickly",
        },
        "icon": "⚡",
        "fake_explanation": (
            "The posting uses urgent or time-pressuring language. "
            "Scammers frequently create artificial urgency to prevent applicants "
            "from doing proper research before applying."
        ),
        "real_explanation": (
            "The posting avoids urgency language, which is typical of legitimate "
            "employers who allow applicants reasonable time to apply."
        ),
    },
    "Vague or Inflated Salary": {
        "keywords": {
            "earn", "earning", "income", "salary", "wage", "pay", "paid",
            "payment", "weekly", "daily", "cash", "money", "hundred",
            "thousand", "per", "hour", "week", "month", "commission",
            "bonus", "unlimited",
        },
        "icon": "💰",
        "fake_explanation": (
            "The posting makes vague or exaggerated financial claims (e.g., "
            "\"earn hundreds per week\"). Fraudulent postings often lead with "
            "income promises rather than specific job details."
        ),
        "real_explanation": (
            "Salary references appear in a structured, specific way — a "
            "hallmark of genuine employers disclosing compensation clearly."
        ),
    },
    "Minimal Requirements / No Experience": {
        "keywords": {
            "experience", "qualification", "degree", "skill", "skills",
            "require", "required", "need", "needed", "background",
            "training", "certificate", "education", "no experience",
            "entry", "beginner", "anyone",
        },
        "icon": "📋",
        "fake_explanation": (
            "The posting offers the role with minimal or no requirements. "
            "Fake postings often promise high pay with no qualifications to "
            "attract as many victims as possible."
        ),
        "real_explanation": (
            "The posting lists specific qualifications or experience, "
            "consistent with a genuine job that has real selection criteria."
        ),
    },
    "Work-from-Home / Remote Claims": {
        "keywords": {
            "home", "remote", "online", "anywhere", "flexible", "work from home",
            "wfh", "telework", "virtual", "location independent",
        },
        "icon": "🏠",
        "fake_explanation": (
            "Heavy emphasis on remote or work-from-home arrangements can be a "
            "red flag when combined with other suspicious signals — "
            "a common pattern in scam job postings."
        ),
        "real_explanation": (
            "Remote-work mentions appear alongside other professional details, "
            "which is typical of legitimate flexible-work roles."
        ),
    },
    "Suspicious Contact Details": {
        "keywords": {
            "gmail", "yahoo", "hotmail", "outlook", "email", "whatsapp",
            "telegram", "contact", "call", "phone", "number", "text",
            "message", "inbox", "apply via",
        },
        "icon": "📧",
        "fake_explanation": (
            "The posting directs applicants to personal email addresses or "
            "messaging apps rather than a corporate recruitement system — "
            "a strong indicator of fraud."
        ),
        "real_explanation": (
            "Contact references appear in a professional context, consistent "
            "with an established organisation's hiring process."
        ),
    },
    "Professional Language & Structure": {
        "keywords": {
            "company", "organisation", "team", "department", "office",
            "headquarters", "benefits", "pension", "healthcare", "holiday",
            "annual", "contract", "permanent", "probation", "reference",
            "interview", "application", "cv", "resume", "role",
            "responsibilities", "candidate",
        },
        "icon": "🏢",
        "fake_explanation": (
            "Despite using professional-sounding words, the overall context "
            "raises concerns. Some scam postings mimic legitimate language "
            "to appear credible."
        ),
        "real_explanation": (
            "The posting uses structured professional language — specific roles, "
            "company references, and formal processes — consistent with a "
            "genuine employer."
        ),
    },
}

_PERSONAL_EMAIL_RE = re.compile(
    r"\b(?:gmail|yahoo|hotmail|outlook|aol|icloud)\.com\b", re.IGNORECASE
)
_SALARY_RE = re.compile(
    r"(?:[£$€]\s?\d[\d,]*|\b\d[\d,]*\s*(?:k|per\s+(?:hour|annum|year|month|week))\b)",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_COMPANY_RE = re.compile(
    r"\b(?:ltd|llc|inc|corp|limited|plc|gmbh|solutions|services|group|global)\b",
    re.IGNORECASE,
)


# Return structural fraud-indicator checks for a given posting
def structural_checklist(title: str, description: str) -> list[dict]:
    text = f"{title} {description}"
    checks = [
        {
            "label": "Company name or identifier present",
            "pass": bool(_COMPANY_RE.search(text) or re.search(
                r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Ltd|LLC|Inc|Corp|Group|Services|Solutions)\b",
                text,
            )),
            "why": (
                "Legitimate employers almost always name the company. "
                "Missing company info is one of the strongest fraud indicators."
            ),
        },
        {
            "label": "Specific salary or pay range stated",
            "pass": bool(_SALARY_RE.search(text)),
            "why": (
                "Genuine postings typically state a salary band. "
                "Vague promises like 'earn big' are a common scam tactic."
            ),
        },
        {
            "label": "Application URL or career portal linked",
            "pass": bool(_URL_RE.search(text)),
            "why": (
                "Reputable employers direct applicants to an official website "
                "or ATS. No URL may indicate an informal or fraudulent listing."
            ),
        },
        {
            "label": "No personal email domain (Gmail/Yahoo/Hotmail)",
            "pass": not bool(_PERSONAL_EMAIL_RE.search(text)),
            "why": (
                "Legitimate companies recruit via corporate email addresses. "
                "Personal email domains are a strong signal of fraud."
            ),
        },
        {
            "label": "Job title present and specific",
            "pass": bool(title and len(title.strip()) >= 5),
            "why": (
                "A clear, specific job title indicates a real role. "
                "Vague titles like 'Opportunity' or missing titles are red flags."
            ),
        },
        {
            "label": "Description is substantive (≥ 200 characters)",
            "pass": len(description.strip()) >= 200,
            "why": (
                "Fake postings are often short and vague. A detailed description "
                "suggests the employer has thought through the actual role."
            ),
        },
    ]
    return checks


# Map SHAP signal tokens to human-readable fraud categories
# Returns categories where at least one token matched, sorted by total impact
def categorise_signals(
    pos_signals: list[dict],
    neg_signals: list[dict],
) -> list[dict]:
    # Build lookup: lowercase token → category name
    token_to_cat: dict[str, str] = {}
    for cat_name, cat in FRAUD_CATEGORIES.items():
        for kw in cat["keywords"]:
            token_to_cat[kw.lower()] = cat_name

    cat_hits: dict[str, dict] = {}  # category_name → {tokens, total_impact, direction}

    for signal in pos_signals:
        token = signal["feature"].lower().strip()
        cat_name = token_to_cat.get(token)
        if cat_name:
            entry = cat_hits.setdefault(cat_name, {"tokens": [], "total_impact": 0.0, "direction": "fraud"})
            entry["tokens"].append(signal["feature"])
            entry["total_impact"] += abs(signal["impact"])

    for signal in neg_signals:
        token = signal["feature"].lower().strip()
        cat_name = token_to_cat.get(token)
        if cat_name:
            entry = cat_hits.setdefault(cat_name, {"tokens": [], "total_impact": 0.0, "direction": "legit"})
            # Keep whichever direction has more total impact
            if entry["direction"] == "fraud":
                if abs(signal["impact"]) > entry["total_impact"]:
                    entry["direction"] = "legit"
                    entry["tokens"] = [signal["feature"]]
                    entry["total_impact"] = abs(signal["impact"])
            else:
                entry["tokens"].append(signal["feature"])
                entry["total_impact"] += abs(signal["impact"])

    result = []
    for cat_name, hit in sorted(cat_hits.items(), key=lambda x: x[1]["total_impact"], reverse=True):
        cat = FRAUD_CATEGORIES[cat_name]
        direction = hit["direction"]
        result.append({
            "name": cat_name,
            "icon": cat["icon"],
            "direction": direction,
            "matched_tokens": hit["tokens"][:5],
            "total_impact": hit["total_impact"],
            "explanation": cat["fake_explanation"] if direction == "fraud" else cat["real_explanation"],
        })
    return result


# Return a plain-English paragraph summarising the model verdict and key reasons
def build_plain_english_summary(
    categorised: list[dict],
    fake_prob: float,
    label: str,
    checklist: list[dict],
) -> str:
    is_fake = label == "fake"
    pct = round(fake_prob * 100)

    fraud_cats = [c for c in categorised if c["direction"] == "fraud"]
    legit_cats = [c for c in categorised if c["direction"] == "legit"]

    failed_checks = [c for c in checklist if not c["pass"]]
    passed_checks = [c for c in checklist if c["pass"]]

    if is_fake:
        intro = (
            f"The model rates this posting as <strong>likely fraudulent</strong> "
            f"with a fraud probability of <strong>{pct}%</strong>. "
        )
        if fraud_cats:
            cat_names = ", ".join(f"<em>{c['name']}</em>" for c in fraud_cats[:3])
            intro += f"The strongest fraud signals detected belong to: {cat_names}. "
        if failed_checks:
            check_labels = ", ".join(f"<em>{c['label'].lower()}</em>" for c in failed_checks[:3])
            intro += (
                f"Structurally, the posting is missing key details: {check_labels} — "
                "all of which are typical of fraudulent listings. "
            )
        intro += (
            "This does not guarantee the posting is fake, but these patterns "
            "are strongly associated with scam job advertisements in the training data."
        )
    else:
        intro = (
            f"The model rates this posting as <strong>likely legitimate</strong> "
            f"with a fraud probability of only <strong>{pct}%</strong>. "
        )
        if legit_cats:
            cat_names = ", ".join(f"<em>{c['name']}</em>" for c in legit_cats[:3])
            intro += f"Legitimacy signals detected include: {cat_names}. "
        if passed_checks:
            check_labels = ", ".join(f"<em>{c['label'].lower()}</em>" for c in passed_checks[:3])
            intro += (
                f"The posting also passes key structural checks: {check_labels}. "
            )
        if fraud_cats:
            intro += (
                "Some ambiguous signals were noted, so always verify the employer "
                "independently before sharing personal information."
            )
        else:
            intro += "Always verify an employer independently before sharing personal information."

    return intro
