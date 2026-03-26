"""
Ablation Study – TextGCN Architecture
=======================================
Tests 6 variants of the TextGCN to empirically justify design choices.

Variants
--------
  1. full          – 3-layer + residual α=0.7  (baseline, window=15)
  2. no_residual   – 3-layer, α=0  (pure graph propagation, no skip connection)
  3. layer_1       – 1-layer GCN
  4. layer_2       – 2-layer GCN
  5. random_edges  – 3-layer, adjacency replaced with random sparse graph
  6. window_5      – 3-layer + residual, but PMI window=5 (very local context)

Each variant is trained with a single seed (42) to keep runtime manageable.
Uses the same hyperparameters as the winning configuration (window=15, lr=3e-3).

Outputs
-------
  reports/ablation_results.csv
"""
import json
import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS_DIR    = ROOT / "reports"

from tuned_models.tune_textgcn import (
    build_pmi_graph, normalize_adj, to_sparse, tokenize,
    best_threshold, eval_at, get_probs,
)

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


# ── Configurable N-layer GCN ─────────────────────────────────────────────────
class NLayerWordGCN(nn.Module):
    """Supports 1, 2, or 3 GCN layers with optional residual blending."""

    def __init__(self, num_words: int, hidden_dim: int = 300,
                 dropout: float = 0.35, residual_alpha: float = 0.7,
                 num_layers: int = 3):
        super().__init__()
        assert 1 <= num_layers <= 3, "num_layers must be 1, 2, or 3"
        self.emb            = nn.Embedding(num_words, hidden_dim)
        nn.init.xavier_uniform_(self.emb.weight)
        self.dropout        = dropout
        self.residual_alpha = residual_alpha
        self.num_layers     = num_layers

        self.lins = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp  = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim // 2, 2)

    def gcn(self, A: torch.Tensor) -> torch.Tensor:
        H0 = self.emb.weight
        H  = H0
        for i, lin in enumerate(self.lins):
            H = F.relu(lin(torch.sparse.mm(A, H)))
            if i < self.num_layers - 1:
                H = F.dropout(H, p=self.dropout, training=self.training)
        # Residual blend: α=0 means pure graph, α=0.7 means 70% graph + 30% raw
        H = (1 - self.residual_alpha) * H0 + self.residual_alpha * H
        return self.norm(H)

    def forward(self, A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        W   = self.gcn(A)
        doc = torch.sparse.mm(X, W) + torch.sparse.mm(X, self.emb.weight)
        doc = F.dropout(doc, p=self.dropout, training=self.training)
        return self.classifier(self.mlp(doc))


def make_random_edges(num_words: int, n_edges: int, seed: int = 42) -> torch.Tensor:
    """Create a random symmetric sparse adjacency (same density as PMI graph)."""
    rng = np.random.default_rng(seed)
    rows, cols = [], []
    for _ in range(n_edges):
        i = int(rng.integers(0, num_words))
        j = int(rng.integers(0, num_words))
        if i != j:
            rows += [i, j]
            cols += [j, i]
    vals = [1.0] * len(rows)
    # Add self-loops
    rows += list(range(num_words))
    cols += list(range(num_words))
    vals += [1.0] * num_words

    idx = torch.tensor([rows, cols], dtype=torch.long)
    val = torch.tensor(vals,         dtype=torch.float32)
    A   = torch.sparse_coo_tensor(idx, val, (num_words, num_words)).coalesce()
    idx, val = A.indices(), A.values()
    deg = torch.zeros(num_words)
    deg.index_add_(0, idx[0], val)
    d   = deg.pow(-0.5).clamp(max=1e6)
    return torch.sparse_coo_tensor(idx, d[idx[0]] * val * d[idx[1]],
                                   (num_words, num_words)).coalesce()


def train_variant(model, A, X_tr, X_va, X_te, X_ob,
                  y_tr, y_va_np, y_te_np,
                  lr=3e-3, weight_decay=1e-5, epochs=200,
                  patience=20, label_smoothing=0.05, device="cpu"):
    """Train one model variant and return test/OB results."""
    torch.manual_seed(42)
    pos = int((torch.tensor(y_tr) == 1).sum())
    neg = int((torch.tensor(y_tr) == 0).sum())
    cw  = torch.tensor([1.0, math.sqrt(neg / max(pos, 1))],
                       dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state, best_vf1, pat = None, -1.0, patience
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(A, X_tr), y_tr_t, weight=cw,
                               label_smoothing=label_smoothing)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        vp  = get_probs(model, A, X_va)
        vf1, _, _ = eval_at(vp, y_va_np)
        if vf1 > best_vf1 + 1e-4:
            best_vf1  = vf1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            pat = patience
        else:
            pat -= 1
            if pat <= 0:
                break

    if best_state:
        model.load_state_dict(best_state)

    val_probs  = get_probs(model, A, X_va)
    test_probs = get_probs(model, A, X_te)
    ob_probs   = get_probs(model, A, X_ob)

    thr       = best_threshold(val_probs, y_va_np)
    tf1, tp, tr = eval_at(test_probs, y_te_np, thr["t"])
    ob_recall  = float(np.mean((ob_probs >= thr["t"]).astype(int) == 1))
    return tf1, tp, tr, ob_recall, thr["t"]


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    print("Loading data...")
    em = pd.read_csv(DATA_PROCESSED / "emscad.csv")
    ob = pd.read_csv(DATA_PROCESSED / "openbay.csv")
    with open(DATA_PROCESSED / "splits.json", encoding="utf-8") as f:
        splits = json.load(f)

    def subset(df, ids):
        return df[df["id"].astype(str).isin(set(map(str, ids)))].copy()

    train_df = subset(em, splits["train_ids"])
    val_df   = subset(em, splits["val_ids"])
    test_df  = subset(em, splits["test_ids"])

    y_tr   = train_df["fraudulent"].astype(int).values
    y_va_np = val_df["fraudulent"].astype(int).values
    y_te_np = test_df["fraudulent"].astype(int).values

    # Common TF-IDF vectorizer (fixed across all ablations)
    vec = TfidfVectorizer(
        tokenizer=tokenize, preprocessor=None, token_pattern=None,
        ngram_range=(1, 3), min_df=2, max_df=0.9,
        sublinear_tf=True, max_features=40_000,
    )
    X_tr_s = vec.fit_transform(train_df["text"])
    X_va_s = vec.transform(val_df["text"])
    X_te_s = vec.transform(test_df["text"])
    X_ob_s = vec.transform(ob["text"])
    vocab     = vec.vocabulary_
    num_words = len(vocab)
    print(f"Vocab: {num_words}")

    X_tr = to_sparse(X_tr_s).to(device)
    X_va = to_sparse(X_va_s).to(device)
    X_te = to_sparse(X_te_s).to(device)
    X_ob = to_sparse(X_ob_s).to(device)

    tok_train = [tokenize(t) for t in train_df["text"].tolist()]

    # Pre-build PMI graphs for window=15 (default) and window=5
    print("Building PMI graph (window=15)...")
    r15, c15, v15, n15 = build_pmi_graph(tok_train, vocab, window_size=15)
    A15 = normalize_adj(r15, c15, v15, n15).to(device)
    n_pmi_edges = len(v15)
    print(f"  PMI edges: {n_pmi_edges}")

    print("Building PMI graph (window=5)...")
    r5, c5, v5, n5 = build_pmi_graph(tok_train, vocab, window_size=5)
    A5  = normalize_adj(r5, c5, v5, n5).to(device)

    # Random adjacency (same density as PMI window=15)
    print("Building random adjacency...")
    A_rand = make_random_edges(num_words, n_pmi_edges // 2).to(device)

    # Reference F1 from existing tuned model report
    TUNED_F1 = 0.8390

    variants = [
        ("full_3layer_w15",       A15,    3, 0.70, "Full model (3-layer, α=0.7, window=15) [REFERENCE]"),
        ("no_residual_3layer_w15",A15,    3, 0.00, "No residual (α=0, 3 layers)"),
        ("1layer_w15",            A15,    1, 0.70, "1-layer GCN"),
        ("2layer_w15",            A15,    2, 0.70, "2-layer GCN"),
        ("random_edges_3layer",   A_rand, 3, 0.70, "Random edges (no PMI)"),
        ("window5_3layer",        A5,     3, 0.70, "PMI window=5"),
    ]

    results = []
    for name, A_use, n_layers, alpha, description in variants:
        print(f"\n[{name}] {description}")
        t0 = time.time()

        model = NLayerWordGCN(
            num_words      = num_words,
            hidden_dim     = 300,
            dropout        = 0.35,
            residual_alpha = alpha,
            num_layers     = n_layers,
        ).to(device)

        tf1, tp, tr, ob_r, thr = train_variant(
            model, A_use, X_tr, X_va, X_te, X_ob,
            y_tr, y_va_np, y_te_np, device=device
        )
        elapsed = time.time() - t0

        results.append({
            "variant":         name,
            "description":     description,
            "emscad_test_f1":  round(tf1, 4),
            "emscad_precision":round(tp,  4),
            "emscad_recall":   round(tr,  4),
            "openbay_recall":  round(ob_r, 4),
            "threshold":       round(thr, 2),
            "delta_vs_full":   round(tf1 - TUNED_F1, 4),
            "elapsed_s":       round(elapsed, 1),
        })
        print(f"  F1={tf1:.4f}  P={tp:.4f}  R={tr:.4f}  OB={ob_r:.4f}  "
              f"thr={thr:.2f}  ({elapsed:.0f}s)")

    df = pd.DataFrame(results)
    out = REPORTS_DIR / "ablation_results.csv"
    df.to_csv(out, index=False)

    print("\n" + "=" * 75)
    print("ABLATION STUDY RESULTS")
    print("=" * 75)
    print(f"{'Variant':<32} {'F1':>7} {'P':>7} {'R':>7} {'OB':>7} {'Δ F1':>8}")
    print("-" * 75)
    for _, row in df.iterrows():
        marker = " ◄" if row["variant"] == "full_3layer_w15" else ""
        print(f"{row['variant']:<32} {row['emscad_test_f1']:>7.4f} "
              f"{row['emscad_precision']:>7.4f} {row['emscad_recall']:>7.4f} "
              f"{row['openbay_recall']:>7.4f} {row['delta_vs_full']:>+8.4f}{marker}")
    print("=" * 75)
    print(f"Tuned ensemble reference F1 = {TUNED_F1:.4f}  "
          f"(single-seed variants are expected to be slightly lower)")
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
