from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, a_norm: torch.Tensor, x_tfidf_sparse: torch.Tensor) -> torch.Tensor:
        word_h = self.gcn(a_norm)
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
    influential_words: List[Dict[str, object]]
    protective_words: List[Dict[str, object]]
    shap_values: object | None = None
    shap_error: str | None = None


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
    # Backward compatibility for artifacts trained from scripts where
    # `tokenize` was saved as `__main__.tokenize` inside the pickled vectorizer.
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

        self.threshold = 0.5
        self._shap_explainer = None
        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path)
            if "threshold" in metrics.columns and not metrics.empty:
                self.threshold = float(metrics.iloc[0]["threshold"])

    def _build_shap_explainer(self):
        if shap is None:
            return None, "SHAP is not installed in the environment."
        if self._shap_explainer is not None:
            return self._shap_explainer, None

        masker = shap.maskers.Text(r"\W+")

        def fake_probability(text_batch):
            probs = self.predict_proba_batch(list(text_batch))
            return probs[:, 1]

        # SHAP may raise when `output_names` is a list but the model has a
        # single output. Let SHAP infer the output layout and rely on the
        # fallback extraction logic in `explain_prediction`.
        self._shap_explainer = shap.Explainer(fake_probability, masker)
        return self._shap_explainer, None

    @staticmethod
    def _scipy_to_torch_sparse(x):
        x = x.tocoo()
        idx = torch.tensor(np.vstack([x.row, x.col]), dtype=torch.long)
        val = torch.tensor(x.data, dtype=torch.float32)
        return torch.sparse_coo_tensor(idx, val, (x.shape[0], x.shape[1])).coalesce()

    def predict_proba_batch(self, texts: List[str]) -> np.ndarray:
        x = self.vectorizer.transform(texts)
        x_t = self._scipy_to_torch_sparse(x).to(self.device)
        with torch.no_grad():
            logits = self.model(self.a_norm, x_t)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    def explain_prediction(self, text: str, top_k: int = 10) -> PredictionResult:
        probs = self.predict_proba_batch([text])[0]
        fake_prob = float(probs[1])
        real_prob = float(probs[0])
        label = "fake" if fake_prob >= self.threshold else "real"
        confidence = fake_prob if label == "fake" else real_prob

        token_counts: Dict[str, int] = {}
        for m in TOKEN_RE.finditer(text):
            token = m.group(0).lower()
            if token in self.vocab:
                token_counts[token] = token_counts.get(token, 0) + 1

        explainer, shap_error = self._build_shap_explainer()
        shap_values = None
        deltas: List[tuple[str, float]] = []
        if explainer is not None:
            try:
                shap_values = explainer([text])
                try:
                    shap_payload = shap_values[:, :, "fake_probability"][0]
                except Exception:
                    shap_payload = shap_values[0]

                token_impacts: Dict[str, float] = {}
                for raw_token, score in zip(shap_payload.data, shap_payload.values):
                    token = str(raw_token).strip().lower()
                    if token in self.vocab:
                        token_impacts[token] = token_impacts.get(token, 0.0) + float(score)

                deltas = list(token_impacts.items())
            except Exception as exc:
                shap_error = f"Failed to compute SHAP explanation: {exc}"

        if not deltas:
            return PredictionResult(
                label,
                fake_prob,
                real_prob,
                confidence,
                self.threshold,
                [],
                [],
                shap_values=shap_values,
                shap_error=shap_error,
            )

        max_abs = max(abs(delta) for _, delta in deltas) if deltas else 1.0

        def payload(token: str, delta: float) -> Dict[str, object]:
            abs_delta = abs(delta)
            occurrence_count = token_counts.get(token, 0)
            if abs_delta >= 0.05:
                impact_strength = "very high"
            elif abs_delta >= 0.02:
                impact_strength = "high"
            else:
                impact_strength = "moderate"

            return {
                "word": token,
                "impact_on_fake_probability": float(delta),
                "absolute_impact": float(abs_delta),
                "normalized_impact": float(delta / max_abs),
                "occurrences": float(occurrence_count),
                "impact_strength": impact_strength,
            }

        influential = sorted((d for d in deltas if d[1] > 0), key=lambda x: x[1], reverse=True)[:top_k]
        protective = sorted((d for d in deltas if d[1] < 0), key=lambda x: x[1])[:top_k]

        return PredictionResult(
            label=label,
            fake_probability=fake_prob,
            real_probability=real_prob,
            confidence=confidence,
            threshold=self.threshold,
            influential_words=[payload(token, delta) for token, delta in influential],
            protective_words=[payload(token, delta) for token, delta in protective],
            shap_values=shap_values,
            shap_error=shap_error,
        )
