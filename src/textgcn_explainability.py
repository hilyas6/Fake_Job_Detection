import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str):
    """Compatibility tokenizer used by persisted TF-IDF vectorizers."""
    if not isinstance(text, str):
        return []
    return TOKEN_RE.findall(text.lower())


class WordGCNPool(nn.Module):
    """Inference-only copy of the TextGCN architecture used in training."""

    def __init__(self, num_words: int, hidden_dim=256, dropout=0.35, residual_alpha=0.7):
        super().__init__()
        self.emb = nn.Embedding(num_words, hidden_dim)

        self.dropout = dropout
        self.residual_alpha = residual_alpha

        self.lin1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, 2)

    def gcn(self, a_norm: torch.Tensor) -> torch.Tensor:
        h0 = self.emb.weight
        h = torch.sparse.mm(a_norm, h0)
        h = self.lin1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = torch.sparse.mm(a_norm, h)
        h = self.lin2(h)
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
class ExplainResult:
    label: str
    fake_probability: float
    real_probability: float
    threshold: float
    influential_words: List[Dict[str, float]]
    protective_words: List[Dict[str, float]]


class TextGCNExplainer:
    def __init__(
        self,
        model_dir: Path = Path("models/textgcn"),
        metrics_path: Path = Path("reports/metrics_textgcn.csv"),
        device: str = "cpu",
    ):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)

        self.vectorizer = joblib.load(self.model_dir / "vectorizer.joblib")
        graph_cache = torch.load(
            self.model_dir / "graph_cache.pt",
            map_location="cpu",
            weights_only=False,
        )
        ckpt = torch.load(
            self.model_dir / "textgcn.pt",
            map_location="cpu",
            weights_only=False,
        )

        self.inv_vocab = graph_cache["inv_vocab"]
        self.vocab = {token: int(idx) for idx, token in self.inv_vocab.items()}

        self.a_norm = torch.sparse_coo_tensor(
            graph_cache["A_norm_indices"],
            graph_cache["A_norm_values"],
            tuple(graph_cache["A_norm_size"]),
        ).coalesce().to(self.device)

        self.model = WordGCNPool(
            num_words=ckpt["num_words"],
            hidden_dim=ckpt["hidden_dim"],
            dropout=ckpt["dropout"],
            residual_alpha=ckpt.get("residual_alpha", 0.7),
        ).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self.threshold = 0.5
        if Path(metrics_path).exists():
            metrics = pd.read_csv(metrics_path)
            if "threshold" in metrics.columns and len(metrics) > 0:
                self.threshold = float(metrics.iloc[0]["threshold"])

    @staticmethod
    def _scipy_to_torch_sparse(x):
        x = x.tocoo()
        idx = torch.tensor(np.vstack([x.row, x.col]), dtype=torch.long)
        val = torch.tensor(x.data, dtype=torch.float32)
        return torch.sparse_coo_tensor(idx, val, (x.shape[0], x.shape[1])).coalesce()

    def predict_proba(self, text: str) -> np.ndarray:
        x = self.vectorizer.transform([text])
        xt = self._scipy_to_torch_sparse(x).to(self.device)
        with torch.no_grad():
            logits = self.model(self.a_norm, xt)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        return probs

    def _unique_known_unigrams(self, text: str) -> List[str]:
        seen = set()
        known = []
        for token in TOKEN_RE.findall(text.lower()):
            if token in seen:
                continue
            seen.add(token)
            if token in self.vocab:
                known.append(token)
        return known

    def explain_text(self, text: str, top_k: int = 10) -> ExplainResult:
        probs = self.predict_proba(text)
        fake_prob = float(probs[1])
        real_prob = float(probs[0])
        label = "fake" if fake_prob >= self.threshold else "real"

        deltas: List[Tuple[str, float]] = []
        for token in self._unique_known_unigrams(text):
            masked = re.sub(rf"\\b{re.escape(token)}\\b", " ", text, flags=re.IGNORECASE)
            masked_probs = self.predict_proba(masked)
            impact = fake_prob - float(masked_probs[1])
            deltas.append((token, impact))

        positive = sorted((d for d in deltas if d[1] > 0), key=lambda x: x[1], reverse=True)[:top_k]
        negative = sorted((d for d in deltas if d[1] < 0), key=lambda x: x[1])[:top_k]

        return ExplainResult(
            label=label,
            fake_probability=fake_prob,
            real_probability=real_prob,
            threshold=self.threshold,
            influential_words=[
                {"word": token, "impact_on_fake_probability": float(delta)} for token, delta in positive
            ],
            protective_words=[
                {"word": token, "impact_on_fake_probability": float(delta)} for token, delta in negative
            ],
        )


def run_demo(explainer: TextGCNExplainer, top_k: int, n_samples: int):
    data_path = Path("data/processed/emscad.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Expected demo dataset at {data_path}")

    df = pd.read_csv(data_path)
    sample = df.sample(n=min(n_samples, len(df)), random_state=42)

    for i, row in sample.iterrows():
        result = explainer.explain_text(str(row["text"]), top_k=top_k)
        print(f"\n--- Sample id={row['id']} (true_label={row['fraudulent']}) ---")
        print(json.dumps(result.__dict__, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Explain TextGCN predictions with token-occlusion.")
    parser.add_argument("--text", type=str, default=None, help="Single job-post text to explain.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top words to show.")
    parser.add_argument("--demo-samples", type=int, default=0, help="Run explanations on random EMSCAD samples.")
    parser.add_argument("--model-dir", type=Path, default=Path("models/textgcn"))
    parser.add_argument("--metrics-path", type=Path, default=Path("reports/metrics_textgcn.csv"))
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    explainer = TextGCNExplainer(
        model_dir=args.model_dir,
        metrics_path=args.metrics_path,
        device=args.device,
    )

    if args.text:
        result = explainer.explain_text(args.text, top_k=args.top_k)
        print(json.dumps(result.__dict__, indent=2))

    if args.demo_samples > 0:
        run_demo(explainer, top_k=args.top_k, n_samples=args.demo_samples)

    if not args.text and args.demo_samples <= 0:
        parser.error("Provide --text for one explanation or --demo-samples for dataset examples.")


if __name__ == "__main__":
    main()
