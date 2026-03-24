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
