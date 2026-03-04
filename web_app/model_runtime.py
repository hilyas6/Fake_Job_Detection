from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st

try:
    import shap
except Exception:  # pragma: no cover - optional dependency handling
    shap = None

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class ImprovedWordGCN(nn.Module):
    """Inference architecture for the improved TextGCN model."""

    def __init__(self, num_words: int, hidden_dim: int = 300, dropout: float = 0.35, residual_alpha: float = 0.7):
        super().__init__()
        self.emb = nn.Embedding(num_words, hidden_dim)
        self.dropout = dropout
        self.residual_alpha = residual_alpha

        self.lin1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lin3 = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim // 2, 2)

    def gcn(self, a_norm: torch.Tensor) -> torch.Tensor:
        h0 = self.emb.weight

        h = torch.sparse.mm(a_norm, h0)
        h = self.lin1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = torch.sparse.mm(a_norm, h)
        h = self.lin2(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = torch.sparse.mm(a_norm, h)
        h = self.lin3(h)
        h = F.relu(h)

        h = (1.0 - self.residual_alpha) * h0 + self.residual_alpha * h
        return self.norm(h)

    def forward_with_cached_word_h(self, x_tfidf_sparse: torch.Tensor, word_h: torch.Tensor) -> torch.Tensor:
        doc_h = torch.sparse.mm(x_tfidf_sparse, word_h)
        doc_h0 = torch.sparse.mm(x_tfidf_sparse, self.emb.weight)
        doc_h = doc_h + doc_h0
        doc_h = F.dropout(doc_h, p=self.dropout, training=self.training)
        doc_h = self.mlp(doc_h)
        return self.classifier(doc_h)


@dataclass
class PredictionResult:
    label: str
    fake_probability: float
    real_probability: float
    confidence: float
    threshold: float


@dataclass
class ExplanationResult:
    top_increase_fake: list[dict[str, float]]
    top_decrease_fake: list[dict[str, float]]
    audit_top_increase_fake: list[dict[str, float]]
    audit_top_decrease_fake: list[dict[str, float]]
    phrase_top_increase_fake: list[dict[str, float]]
    phrase_top_decrease_fake: list[dict[str, float]]
    stability: dict[str, float] | None = None
    shap_values: Any | None = None
    shap_error: str | None = None
    mode: str = "fast"


def _is_git_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as f:
            first = f.readline().strip()
        return first.startswith("version https://git-lfs.github.com/spec/v1")
    except (OSError, UnicodeDecodeError):
        return False


def _load_joblib(path: Path):
    if _is_git_lfs_pointer(path):
        raise RuntimeError(f"{path} is a Git LFS pointer. Run `git lfs pull` and retry.")
    main_module = sys.modules.get("__main__")
    if main_module is not None and not hasattr(main_module, "tokenize"):
        setattr(main_module, "tokenize", tokenize)
    return joblib.load(path)


def tokenize(text: str):
    if not isinstance(text, str):
        return []
    return TOKEN_RE.findall(text.lower())


class ImprovedTextGCNService:
    def __init__(
        self,
        model_dir: Path = Path("models/textgcn"),
        metrics_path: Path = Path("reports/metrics_textgcn_improved.csv"),
        device: str = "cpu",
    ):
        self.model_dir = model_dir
        self.device = torch.device(device)

        self.vectorizer = _load_joblib(model_dir / "vectorizer_improved.joblib")
        graph_cache = torch.load(model_dir / "graph_cache_improved.pt", map_location="cpu", weights_only=False)
        ckpt = torch.load(model_dir / "textgcn_improved.pt", map_location="cpu", weights_only=False)

        self.inv_vocab = graph_cache["inv_vocab"]
        self.vocab = {token: int(idx) for idx, token in self.inv_vocab.items()}

        self.a_norm = torch.sparse_coo_tensor(
            graph_cache["A_norm_indices"],
            graph_cache["A_norm_values"],
            tuple(graph_cache["A_norm_size"]),
        ).coalesce().to(self.device)

        self.model = ImprovedWordGCN(
            num_words=int(ckpt["num_words"]),
            hidden_dim=int(ckpt["hidden_dim"]),
            dropout=float(ckpt["dropout"]),
            residual_alpha=float(ckpt.get("residual_alpha", 0.7)),
        ).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        with torch.no_grad():
            self._cached_word_h = self.model.gcn(self.a_norm)

        self.threshold = 0.5
        self._shap_explainer = None
        self._shap_cache: dict[tuple[str, str], object] = {}
        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path)
            if "threshold" in metrics.columns and not metrics.empty:
                self.threshold = float(metrics.iloc[0]["threshold"])

    def preprocess_text(self, text: str):
        return self.vectorizer.transform([text])

    @staticmethod
    def _scipy_to_torch_sparse(x):
        x = x.tocoo()
        idx = torch.tensor(np.vstack([x.row, x.col]), dtype=torch.long)
        val = torch.tensor(x.data, dtype=torch.float32)
        return torch.sparse_coo_tensor(idx, val, (x.shape[0], x.shape[1])).coalesce()

    def predict_from_preprocessed(self, x) -> PredictionResult:
        x_t = self._scipy_to_torch_sparse(x).to(self.device)
        with torch.no_grad():
            logits = self.model.forward_with_cached_word_h(x_t, self._cached_word_h)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        fake_prob = float(probs[1])
        real_prob = float(probs[0])
        label = "fake" if fake_prob >= self.threshold else "real"
        confidence = fake_prob if label == "fake" else real_prob
        return PredictionResult(
            label=label,
            fake_probability=fake_prob,
            real_probability=real_prob,
            confidence=confidence,
            threshold=self.threshold,
        )

    def _build_shap_explainer(self):
        if shap is None:
            return None, "SHAP is not installed in the environment."
        if self._shap_explainer is not None:
            return self._shap_explainer, None

        masker = shap.maskers.Text(r"\W+")

        def fake_probability(text_batch):
            x = self.vectorizer.transform(list(text_batch))
            x_t = self._scipy_to_torch_sparse(x).to(self.device)
            with torch.no_grad():
                logits = self.model.forward_with_cached_word_h(x_t, self._cached_word_h)
                probs = F.softmax(logits, dim=1).cpu().numpy()
            return probs[:, 1]

        self._shap_explainer = shap.Explainer(fake_probability, masker)
        return self._shap_explainer, None

    @staticmethod
    def _extract_shap_token_items(sample: Any) -> list[tuple[str, float]]:
        """Extract token-level SHAP items across common payload shapes."""
        values = np.asarray(getattr(sample, "values", []))
        data = np.asarray(getattr(sample, "data", []), dtype=object)

        if values.ndim > 1:
            values = values.reshape(-1)
        if data.ndim > 1:
            data = data.reshape(-1)

        token_items: list[tuple[str, float]] = []
        for token, value in zip(data, values):
            token_str = str(token).strip()
            if not token_str:
                continue
            token_items.append((token_str, float(value)))
        return token_items

    def _sorted_feature_candidates(self, text: str, max_features: int = 30) -> list[tuple[str, float]]:
        x = self.vectorizer.transform([text])
        row = x.tocoo()
        feature_names = self.vectorizer.get_feature_names_out()
        candidates = [(str(feature_names[col]), float(val)) for col, val in zip(row.col, row.data)]
        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[:max_features]

    def _mask_phrase(self, text: str, phrase: str) -> str:
        pattern = re.compile(rf"\b{re.escape(phrase)}\b", flags=re.IGNORECASE)
        return pattern.sub(" ", text)

    def _occlusion_audit(self, text: str, fake_prob: float, top_k: int = 10) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
        candidates = self._sorted_feature_candidates(text)
        if not candidates:
            return [], []

        masked_texts = [self._mask_phrase(text, phrase) for phrase, _ in candidates]
        x = self.vectorizer.transform(masked_texts)
        x_t = self._scipy_to_torch_sparse(x).to(self.device)
        with torch.no_grad():
            logits = self.model.forward_with_cached_word_h(x_t, self._cached_word_h)
            probs = F.softmax(logits, dim=1).cpu().numpy()

        deltas = []
        for (phrase, tfidf_weight), row in zip(candidates, probs):
            delta = fake_prob - float(row[1])
            deltas.append((phrase, delta, tfidf_weight))

        positives = [
            {"feature": feature, "impact": impact, "tfidf_weight": tfidf}
            for feature, impact, tfidf in sorted((d for d in deltas if d[1] > 0), key=lambda x: x[1], reverse=True)[:top_k]
        ]
        negatives = [
            {"feature": feature, "impact": impact, "tfidf_weight": tfidf}
            for feature, impact, tfidf in sorted((d for d in deltas if d[1] < 0), key=lambda x: x[1])[:top_k]
        ]
        return positives, negatives

    def _split_phrase_vs_token(self, records: list[dict[str, float]]) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
        phrase = [item for item in records if " " in item["feature"]]
        token = [item for item in records if " " not in item["feature"]]
        return token, phrase

    @staticmethod
    def _rank_biased_overlap(a: list[str], b: list[str], k: int = 10, p: float = 0.9) -> float:
        if not a or not b:
            return 0.0
        score = 0.0
        seen_a, seen_b = set(), set()
        depth = min(k, max(len(a), len(b)))
        for d in range(1, depth + 1):
            if d <= len(a):
                seen_a.add(a[d - 1])
            if d <= len(b):
                seen_b.add(b[d - 1])
            overlap = len(seen_a.intersection(seen_b)) / d
            score += overlap * (p ** (d - 1))
        return (1 - p) * score

    def _stability_probe(self, text: str, baseline_top: list[dict[str, float]]) -> dict[str, float]:
        if not baseline_top:
            return {"rbo_top10": 0.0}
        perturbed = " ".join(text.split())
        if perturbed == text:
            perturbed = f" {text} "
        alt = self.explain_text(perturbed, mode="fast")
        base = [item["feature"] for item in baseline_top[:10]]
        comp = [item["feature"] for item in alt.top_increase_fake[:10]]
        return {"rbo_top10": self._rank_biased_overlap(base, comp, k=10)}

    def explain_text(self, text: str, mode: str = "fast") -> ExplanationResult:
        explainer, shap_error = self._build_shap_explainer()
        if explainer is None:
            return ExplanationResult([], [], [], [], [], [], shap_values=None, shap_error=shap_error, mode=mode)

        cache_key = (text, mode)
        try:
            if cache_key in self._shap_cache:
                shap_values = self._shap_cache[cache_key]
            else:
                max_evals = 100 if mode == "fast" else 250
                shap_values = explainer([text], max_evals=max_evals)
                self._shap_cache[cache_key] = shap_values
        except Exception as exc:
            return ExplanationResult([], [], [], [], [], [], shap_values=None, shap_error=f"Failed to compute SHAP explanation: {exc}", mode=mode)

        sample = shap_values[0]
        token_items = self._extract_shap_token_items(sample)

        aggregated: dict[str, float] = {}
        for token, value in token_items:
            aggregated[token] = aggregated.get(token, 0.0) + value

        positives = sorted(((k, v) for k, v in aggregated.items() if v > 0), key=lambda p: p[1], reverse=True)
        negatives = sorted(((k, v) for k, v in aggregated.items() if v < 0), key=lambda p: p[1])
        top_k = 10 if mode == "fast" else 20

        top_increase_fake = [{"feature": token, "impact": value} for token, value in positives[:top_k]]
        top_decrease_fake = [{"feature": token, "impact": value} for token, value in negatives[:top_k]]
        audit_pos, audit_neg = self._occlusion_audit(text, fake_prob=self.predict_from_preprocessed(self.preprocess_text(text)).fake_probability, top_k=top_k)

        _, phrase_top_increase_fake = self._split_phrase_vs_token(audit_pos)
        _, phrase_top_decrease_fake = self._split_phrase_vs_token(audit_neg)

        stability = self._stability_probe(text, top_increase_fake) if mode == "full" else None

        return ExplanationResult(
            top_increase_fake=top_increase_fake,
            top_decrease_fake=top_decrease_fake,
            audit_top_increase_fake=audit_pos,
            audit_top_decrease_fake=audit_neg,
            phrase_top_increase_fake=phrase_top_increase_fake,
            phrase_top_decrease_fake=phrase_top_decrease_fake,
            stability=stability,
            shap_values=shap_values,
            shap_error=None,
            mode=mode,
        )


@st.cache_resource(show_spinner=True)
def load_model() -> ImprovedTextGCNService:
    return ImprovedTextGCNService(
        model_dir=Path("models/textgcn"),
        metrics_path=Path("reports/metrics_textgcn_improved.csv"),
    )
