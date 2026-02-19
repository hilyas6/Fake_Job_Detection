import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _is_git_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        return first_line.startswith("version https://git-lfs.github.com/spec/v1")
    except (OSError, UnicodeDecodeError):
        return False


def _load_joblib_or_raise(path: Path):
    if _is_git_lfs_pointer(path):
        raise RuntimeError(
            f"Model artifact {path} is a Git LFS pointer, not the real file. "
            "Run `git lfs pull` to download model artifacts, then retry."
        )
    return joblib.load(path)


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
    confidence: float
    text_stats: Dict[str, int]
    influential_words: List[Dict[str, object]]
    protective_words: List[Dict[str, object]]


class TextGCNExplainer:
    def __init__(
        self,
        model_dir: Path = Path("models/textgcn"),
        metrics_path: Path = Path("reports/metrics_textgcn.csv"),
        device: str = "cpu",
    ):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)

        self.vectorizer = _load_joblib_or_raise(self.model_dir / "vectorizer.joblib")
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
        probs = self.predict_proba_batch([text])
        return probs[0]

    def predict_proba_batch(self, texts: List[str]) -> np.ndarray:
        x = self.vectorizer.transform(texts)
        xt = self._scipy_to_torch_sparse(x).to(self.device)
        with torch.no_grad():
            logits = self.model(self.a_norm, xt)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    def _token_spans(self, text: str) -> Dict[str, List[Tuple[int, int]]]:
        spans: Dict[str, List[Tuple[int, int]]] = {}
        for m in TOKEN_RE.finditer(text):
            token = m.group(0).lower()
            if token not in self.vocab:
                continue
            spans.setdefault(token, []).append((m.start(), m.end()))
        return spans

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
        confidence = fake_prob if label == "fake" else real_prob

        token_spans = self._token_spans(text)
        known_tokens = list(token_spans.keys())

        if not known_tokens:
            return ExplainResult(
                label=label,
                fake_probability=fake_prob,
                real_probability=real_prob,
                threshold=self.threshold,
                confidence=confidence,
                text_stats={"all_tokens": len(TOKEN_RE.findall(text)), "known_tokens": 0},
                influential_words=[],
                protective_words=[],
            )

        masked_texts = [
            re.sub(rf"\b{re.escape(token)}\b", " ", text, flags=re.IGNORECASE)
            for token in known_tokens
        ]
        masked_probs = self.predict_proba_batch(masked_texts)

        deltas: List[Tuple[str, float]] = []
        for token, token_probs in zip(known_tokens, masked_probs):
            impact = fake_prob - float(token_probs[1])
            deltas.append((token, impact))

        max_abs_impact = max(abs(delta) for _, delta in deltas) if deltas else 1.0

        def to_payload(token: str, delta: float) -> Dict[str, float]:
            return {
                "word": token,
                "impact_on_fake_probability": float(delta),
                "normalized_impact": float(delta / max_abs_impact),
                "occurrences": len(token_spans[token]),
                "spans": [
                    {"start": int(start), "end": int(end)}
                    for start, end in token_spans[token]
                ],
            }

        positive = sorted((d for d in deltas if d[1] > 0), key=lambda x: x[1], reverse=True)[:top_k]
        negative = sorted((d for d in deltas if d[1] < 0), key=lambda x: x[1])[:top_k]

        return ExplainResult(
            label=label,
            fake_probability=fake_prob,
            real_probability=real_prob,
            threshold=self.threshold,
            confidence=confidence,
            text_stats={"all_tokens": len(TOKEN_RE.findall(text)), "known_tokens": len(known_tokens)},
            influential_words=[
                to_payload(token, delta) for token, delta in positive
            ],
            protective_words=[
                to_payload(token, delta) for token, delta in negative
            ],
        )


def _format_word_list(words: List[Dict[str, object]], heading: str) -> List[str]:
    if not words:
        return [f"{heading}: none"]

    lines = [heading + ":"]
    for idx, word_data in enumerate(words, start=1):
        token = str(word_data["word"])
        impact = float(word_data["impact_on_fake_probability"])
        norm = float(word_data["normalized_impact"])
        occurrences = int(word_data["occurrences"])
        lines.append(
            f"  {idx:>2}. {token:<18} impact={impact:+.4f} "
            f"normalized={norm:+.2f} occurrences={occurrences}"
        )
    return lines


def format_explain_result(result: ExplainResult) -> str:
    lines = [
        "TextGCN Explainability Summary",
        "=" * 32,
        f"Prediction : {result.label.upper()} (confidence={result.confidence:.2%})",
        (
            "Probabilities : "
            f"fake={result.fake_probability:.2%} | real={result.real_probability:.2%} "
            f"(threshold={result.threshold:.2f})"
        ),
        (
            "Coverage : "
            f"known tokens={result.text_stats['known_tokens']} / "
            f"all tokens={result.text_stats['all_tokens']}"
        ),
        "",
    ]

    lines.extend(
        _format_word_list(
            result.influential_words,
            "Top words pushing prediction toward FAKE",
        )
    )
    lines.append("")
    lines.extend(
        _format_word_list(
            result.protective_words,
            "Top words pushing prediction toward REAL",
        )
    )
    return "\n".join(lines)


def run_demo(explainer: TextGCNExplainer, top_k: int, n_samples: int):
    data_path = Path("data/processed/emscad.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Expected demo dataset at {data_path}")

    df = pd.read_csv(data_path)
    sample = df.sample(n=min(n_samples, len(df)), random_state=42)

    for i, row in sample.iterrows():
        result = explainer.explain_text(str(row["text"]), top_k=top_k)
        print(f"\n--- Sample id={row['id']} (true_label={row['fraudulent']}) ---")
        print(format_explain_result(result))


def run_interactive(explainer: TextGCNExplainer, top_k: int, output_format: str):
    print("Interactive TextGCN explainability mode")
    print("Paste a job posting (single line) and press enter.")
    print("Type 'exit' or 'quit' to finish.\n")

    while True:
        try:
            text = input("job-post> ").strip()
        except EOFError:
            print("\nInput stream closed. Exiting interactive mode.")
            break

        if not text:
            continue

        if text.lower() in {"exit", "quit"}:
            print("Exiting interactive mode.")
            break

        result = explainer.explain_text(text, top_k=top_k)
        if output_format == "json":
            print(json.dumps(asdict(result), indent=2))
        else:
            print(format_explain_result(result))
        print()


def main():
    parser = argparse.ArgumentParser(description="Explain TextGCN predictions with token-occlusion.")
    parser.add_argument("--text", type=str, default=None, help="Single job-post text to explain.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top words to show.")
    parser.add_argument("--demo-samples", type=int, default=0, help="Run explanations on random EMSCAD samples.")
    parser.add_argument("--interactive", action="store_true", help="Launch prompt for repeated custom inputs.")
    parser.add_argument("--model-dir", type=Path, default=Path("models/textgcn"))
    parser.add_argument("--metrics-path", type=Path, default=Path("reports/metrics_textgcn.csv"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output-format",
        choices=["readable", "json"],
        default="readable",
        help="Display a concise human-readable summary or raw JSON output.",
    )
    args = parser.parse_args()

    explainer = TextGCNExplainer(
        model_dir=args.model_dir,
        metrics_path=args.metrics_path,
        device=args.device,
    )

    if args.text:
        result = explainer.explain_text(args.text, top_k=args.top_k)
        if args.output_format == "json":
            print(json.dumps(asdict(result), indent=2))
        else:
            print(format_explain_result(result))

    if args.demo_samples > 0:
        run_demo(explainer, top_k=args.top_k, n_samples=args.demo_samples)

    if args.interactive:
        run_interactive(explainer, top_k=args.top_k, output_format=args.output_format)

    if not args.text and args.demo_samples <= 0 and not args.interactive:
        parser.error("Provide --text, --demo-samples, or --interactive.")


if __name__ == "__main__":
    main()
