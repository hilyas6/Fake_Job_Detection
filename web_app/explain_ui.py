from __future__ import annotations

import html
import re
from typing import Any


def normalize_term(term: str) -> str:
    return re.sub(r"\s+", " ", (term or "").strip().lower())


def bucket_strength(value: float, thresholds: tuple[float, float]) -> str:
    low, high = thresholds
    abs_value = abs(value)
    if abs_value >= high:
        return "High"
    if abs_value >= low:
        return "Medium"
    return "Low"


def extract_evidence_snippets(text: str, term: str, max_hits: int = 2, window: int = 60) -> list[str]:
    clean_term = normalize_term(term)
    if not clean_term or not text:
        return []

    snippets: list[str] = []
    for match in re.finditer(re.escape(clean_term), text.lower()):
        start = max(0, match.start() - window)
        end = min(len(text), match.end() + window)
        snippet = text[start:end].strip()
        if snippet:
            snippets.append(f'"{snippet}"')
        if len(snippets) >= max_hits:
            break
    return snippets


def build_reason_templates() -> list[dict[str, Any]]:
    return [
        {
            "id": "off_platform",
            "title": "Off-platform contact request",
            "explanation": "Contact instructions outside official channels can increase fraud risk.",
            "direction": "pushes_fake",
            "match_terms": ["whatsapp", "telegram", "signal", "wechat", "text me", "dm"],
        },
        {
            "id": "upfront_payment",
            "title": "Upfront payment or fee language",
            "explanation": "Requests for money during hiring are a common risk pattern.",
            "direction": "pushes_fake",
            "match_terms": ["fee", "deposit", "pay to apply", "training fee", "upfront"],
        },
        {
            "id": "sensitive_info",
            "title": "Sensitive personal data request",
            "explanation": "Early requests for identity or banking details may indicate abuse risk.",
            "direction": "pushes_fake",
            "match_terms": ["bank", "routing", "ssn", "passport", "id photo", "national insurance"],
        },
        {
            "id": "urgency_guarantee",
            "title": "Unrealistic urgency or guarantee",
            "explanation": "Guarantees and pressure tactics can increase scam likelihood.",
            "direction": "pushes_fake",
            "match_terms": ["immediate hire", "guaranteed", "no interview", "limited spots", "urgent"],
        },
        {
            "id": "suspicious_comp",
            "title": "Suspicious compensation claims",
            "explanation": "Unusually high pay claims without detail can increase risk.",
            "direction": "pushes_fake",
            "match_terms": ["$800 daily", "earn per day", "too good to be true", "guaranteed income"],
        },
        {
            "id": "vague_company",
            "title": "Vague company identity",
            "explanation": "Unclear company identity can reduce trust in posting authenticity.",
            "direction": "pushes_fake",
            "match_terms": ["our company", "leading company", "global company"],
        },
        {
            "id": "real_signals",
            "title": "Structured hiring details",
            "explanation": "Concrete role details and formal process language often aligns with legitimate postings.",
            "direction": "pushes_real",
            "match_terms": [
                "structured interview",
                "benefits",
                "responsibilities",
                "requirements",
                "official apply link",
                "company.com",
            ],
        },
    ]


def map_features_to_reasons(text: str, phrase_impacts: list[dict[str, Any]], token_impacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    signals = phrase_impacts if phrase_impacts else token_impacts
    if not signals:
        return []

    normalized_signals = [
        {
            "feature": normalize_term(str(item.get("feature", ""))),
            "raw_feature": str(item.get("feature", "")),
            "impact": float(item.get("impact", 0.0)),
        }
        for item in signals
        if str(item.get("feature", "")).strip()
    ]
    templates = build_reason_templates()
    reasons: list[dict[str, Any]] = []

    for template in templates:
        matched = []
        score = 0.0
        for signal in normalized_signals:
            for term in template["match_terms"]:
                norm_term = normalize_term(term)
                if norm_term and norm_term in signal["feature"]:
                    matched.append(signal["raw_feature"])
                    score += signal["impact"]
                    break

        if template["id"] == "vague_company" and "our company" in text.lower():
            if not re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", text):
                score += 0.03
                matched.append("our company")

        if not matched:
            continue

        evidence = []
        for term in dict.fromkeys(matched):
            evidence.extend(extract_evidence_snippets(text, term, max_hits=2))

        direction = template["direction"]
        if score < 0:
            direction = "pushes_real"
        elif score > 0:
            direction = "pushes_fake"

        reasons.append(
            {
                "id": template["id"],
                "title": template["title"],
                "explanation": template["explanation"],
                "direction": direction,
                "score": score,
                "matched_terms": list(dict.fromkeys(matched)),
                "evidence_snippets": evidence[:3],
            }
        )

    reasons.sort(key=lambda item: abs(item["score"]), reverse=True)
    reasons = reasons[:7]
    if not reasons:
        return []

    abs_scores = [abs(r["score"]) for r in reasons]
    q1, q2 = _quantile_thresholds(abs_scores)
    for reason in reasons:
        reason["strength"] = bucket_strength(reason["score"], (q1, q2))
    return reasons


def _quantile_thresholds(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    sorted_vals = sorted(values)

    def pick(q: float) -> float:
        if len(sorted_vals) == 1:
            return sorted_vals[0]
        idx = int(round((len(sorted_vals) - 1) * q))
        return sorted_vals[max(0, min(idx, len(sorted_vals) - 1))]

    return pick(0.33), pick(0.66)


def find_spans(text: str, term: str, max_hits: int = 3) -> list[tuple[int, int]]:
    if not text or not term:
        return []
    term = term.strip()
    if not term:
        return []

    is_alnum = bool(re.fullmatch(r"[\w\s]+", term))
    pattern = rf"\b{re.escape(term)}\b" if is_alnum else re.escape(term)
    spans = []
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        spans.append((match.start(), match.end()))
        if len(spans) >= max_hits:
            break
    return spans


def build_highlight_spans(text: str, reasons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for reason in reasons:
        kind = "fake" if reason.get("direction") == "pushes_fake" else "real"
        for term in reason.get("matched_terms", []):
            for start, end in find_spans(text, term):
                spans.append({"start": start, "end": end, "kind": kind, "label": reason.get("title", "Reason")})

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
            continue
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
        frag = html.escape(text[start:end])
        color = "#ffd6d6" if span["kind"] == "fake" else "#d8f5d0"
        label = html.escape(str(span.get("label", "signal")))
        parts.append(f"<span title='{label}' style='background:{color}; padding:0 2px; border-radius:3px;'>{frag}</span>")
        cursor = end
    if cursor < len(text):
        parts.append(html.escape(text[cursor:]))
    return "<div style='white-space:pre-wrap; line-height:1.6'>" + "".join(parts) + "</div>"


def bucket_magnitude(values: list[float]) -> list[str]:
    if not values:
        return []
    abs_values = [abs(v) for v in values]
    q1, q2 = _quantile_thresholds(abs_values)
    return [bucket_strength(v, (q1, q2)).replace("High", "Large").replace("Medium", "Medium").replace("Low", "Small") for v in values]


def build_counterfactual_checks(reasons: list[dict[str, Any]]) -> list[str]:
    by_id = {r.get("id") for r in reasons}
    checks: list[str] = []
    if "off_platform" in by_id:
        checks.append("If official company-domain contact details were verified, risk might decrease.")
    if "upfront_payment" in by_id:
        checks.append("If no hiring fees are required and HR confirms this on the official site, risk might decrease.")
    if "sensitive_info" in by_id:
        checks.append("If employer identity is confirmed before sharing bank or ID data, risk might decrease.")
    if "suspicious_comp" in by_id:
        checks.append("If compensation matches market ranges and appears on the company careers page, risk might decrease.")
    if "vague_company" in by_id:
        checks.append("If a verifiable legal company name and official careers link are provided, risk might decrease.")
    if not checks:
        checks.append("If role details are confirmed on an official company careers page, risk might decrease.")
    return checks[:5]


def redact_emails(text: str) -> str:
    return re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[REDACTED_EMAIL]", text)


def redact_phones(text: str) -> str:
    return re.sub(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)", "[REDACTED_PHONE]", text)


def redact_urls(text: str) -> str:
    return re.sub(r"(?:https?://\S+|www\.\S+)", "[REDACTED_URL]", text)
