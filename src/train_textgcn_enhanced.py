# src/train_textgcn_enhanced.py
import json
import math
import re
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

from src.config import PATHS, CFG

# -----------------------------
# Tokenization
# -----------------------------
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str):
    if not isinstance(text, str):
        return []
    return TOKEN_RE.findall(text.lower())


# -----------------------------
# Build PMI graph
# -----------------------------
def build_pmi_graph(tokenized_docs, vocab_index, window_size=20, pmi_threshold=0.0):
    word_window_count = Counter()
    pair_count = Counter()
    total_windows = 0

    for tokens in tokenized_docs:
        ids = [vocab_index[t] for t in tokens if t in vocab_index]
        if not ids:
            continue

        if len(ids) <= window_size:
            windows = [ids]
        else:
            windows = [ids[i:i + window_size] for i in range(len(ids) - window_size + 1)]

        for w in windows:
            total_windows += 1
            unique = set(w)
            for i in unique:
                word_window_count[i] += 1
            unique_list = list(unique)
            for i in range(len(unique_list)):
                for j in range(i + 1, len(unique_list)):
                    a, b = unique_list[i], unique_list[j]
                    if a < b:
                        pair_count[(a, b)] += 1
                    else:
                        pair_count[(b, a)] += 1

    rows = []
    cols = []
    vals = []

    for (i, j), cij in pair_count.items():
        ci = word_window_count[i]
        cj = word_window_count[j]
        pmi = math.log((cij * total_windows) / (ci * cj + 1e-12) + 1e-12)
        if pmi > pmi_threshold:
            rows += [i, j]
            cols += [j, i]
            vals += [pmi, pmi]

    num_words = len(vocab_index)
    return rows, cols, vals, num_words


def normalize_sparse_adj(rows, cols, vals, n):
    rows = rows + list(range(n))
    cols = cols + list(range(n))
    vals = vals + [1.0] * n

    idx = torch.tensor([rows, cols], dtype=torch.long)
    val = torch.tensor(vals, dtype=torch.float32)

    A = torch.sparse_coo_tensor(idx, val, (n, n)).coalesce()
    idx = A.indices()
    val = A.values()

    deg = torch.zeros(n, dtype=torch.float32)
    deg.index_add_(0, idx[0], val)
    deg_inv_sqrt = torch.pow(deg + 1e-12, -0.5)

    val_norm = deg_inv_sqrt[idx[0]] * val * deg_inv_sqrt[idx[1]]
    A_norm = torch.sparse_coo_tensor(idx, val_norm, (n, n)).coalesce()
    return A_norm


# -----------------------------
# Enhanced Model - BASED ON YOUR WORKING ORIGINAL
# -----------------------------
class ImprovedWordGCN(nn.Module):
    """
    Based on your original working model with strategic improvements:
    1. One extra GCN layer (3 instead of 2)
    2. Slightly larger hidden dim
    3. Better MLP classifier
    """

    def __init__(self, num_words: int, hidden_dim=300, dropout=0.35, residual_alpha=0.7):
        super().__init__()
        self.emb = nn.Embedding(num_words, hidden_dim)
        nn.init.xavier_uniform_(self.emb.weight)

        self.dropout = dropout
        self.residual_alpha = residual_alpha

        # Three GCN layers (original had 2)
        self.lin1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lin3 = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.norm = nn.LayerNorm(hidden_dim)

        # Improved MLP (original had 1 layer, now 2)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim // 2, 2)

    def gcn(self, A_norm):
        H0 = self.emb.weight  # (V, d)

        # Layer 1
        H = torch.sparse.mm(A_norm, H0)
        H = self.lin1(H)
        H = F.relu(H)
        H = F.dropout(H, p=self.dropout, training=self.training)

        # Layer 2
        H = torch.sparse.mm(A_norm, H)
        H = self.lin2(H)
        H = F.relu(H)
        H = F.dropout(H, p=self.dropout, training=self.training)

        # Layer 3 (NEW)
        H = torch.sparse.mm(A_norm, H)
        H = self.lin3(H)
        H = F.relu(H)

        # Residual connection
        H = (1.0 - self.residual_alpha) * H0 + self.residual_alpha * H
        return self.norm(H)  # (V, d)

    def forward(self, A_norm, X_tfidf_sparse):
        word_H = self.gcn(A_norm)  # (V, d)
        doc_H = torch.sparse.mm(X_tfidf_sparse, word_H)  # (N, d)
        doc_H0 = torch.sparse.mm(X_tfidf_sparse, self.emb.weight)  # (N, d)
        doc_H = doc_H + doc_H0
        doc_H = F.dropout(doc_H, p=self.dropout, training=self.training)
        doc_H = self.mlp(doc_H)
        logits = self.classifier(doc_H)  # (N, 2)
        return logits


# -----------------------------
# Config - CLOSE TO ORIGINAL
# -----------------------------
@dataclass
class TrainConfig:
    hidden_dim: int = 300  # Slightly larger than original 256
    dropout: float = 0.35
    lr: float = 3e-3
    weight_decay: float = 1e-5
    epochs: int = 100
    patience: int = 15
    label_smoothing: float = 0.05

    window_size: int = 20
    pmi_threshold: float = 0.0
    max_features: int = 40000
    residual_alpha: float = 0.7


# -----------------------------
# Utilities
# -----------------------------
def load_splits():
    with open(PATHS.data_processed / "splits.json", "r", encoding="utf-8") as f:
        return json.load(f)


def subset_by_ids(df: pd.DataFrame, ids):
    ids = set(map(str, ids))
    return df[df["id"].astype(str).isin(ids)].copy()


def scipy_to_torch_sparse(X):
    X = X.tocoo()
    idx = torch.tensor(np.vstack([X.row, X.col]), dtype=torch.long)
    val = torch.tensor(X.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, (X.shape[0], X.shape[1])).coalesce()


@torch.no_grad()
def eval_split(model, A_norm, X, y_true, threshold=0.5):
    model.eval()
    logits = model(A_norm, X)
    probs = F.softmax(logits, dim=1)[:, 1]
    pred = (probs >= threshold).to(torch.long).cpu().numpy()
    y = y_true.cpu().numpy()
    f1 = f1_score(y, pred, zero_division=0)
    prec = precision_score(y, pred, zero_division=0)
    rec = recall_score(y, pred, zero_division=0)
    return f1, prec, rec


@torch.no_grad()
def find_best_threshold(model, A_norm, X, y_true, grid=None):
    if grid is None:
        grid = np.linspace(0.1, 0.9, 41)
    best = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for t in grid:
        f1, prec, rec = eval_split(model, A_norm, X, y_true, threshold=float(t))
        if f1 > best["f1"] + 1e-6:
            best = {"threshold": float(t), "f1": f1, "precision": prec, "recall": rec}
    return best


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig()

    PATHS.models_textgcn.mkdir(parents=True, exist_ok=True)
    PATHS.reports.mkdir(parents=True, exist_ok=True)

    em = pd.read_csv(PATHS.data_processed / "emscad.csv")
    ob = pd.read_csv(PATHS.data_processed / "openbay.csv")
    splits = load_splits()

    train_df = subset_by_ids(em, splits["train_ids"])
    val_df = subset_by_ids(em, splits["val_ids"])
    test_df = subset_by_ids(em, splits["test_ids"])

    # TF-IDF - with trigrams for better pattern capture
    vec = TfidfVectorizer(
        tokenizer=tokenize,
        preprocessor=None,
        token_pattern=None,
        ngram_range=(1, 3),  # IMPROVED: trigrams instead of bigrams
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        max_features=cfg.max_features
    )

    X_train_s = vec.fit_transform(train_df["text"])
    X_val_s = vec.transform(val_df["text"])
    X_test_s = vec.transform(test_df["text"])
    X_ob_s = vec.transform(ob["text"])

    vocab = vec.vocabulary_
    inv_vocab = {i: t for t, i in vocab.items()}
    num_words = len(vocab)
    print(f"Vocab size: {num_words}")

    # PMI graph
    tokenized_train = [tokenize(t) for t in train_df["text"].tolist()]
    rows, cols, vals, n = build_pmi_graph(
        tokenized_train, vocab,
        window_size=cfg.window_size,
        pmi_threshold=cfg.pmi_threshold
    )
    A_norm = normalize_sparse_adj(rows, cols, vals, n).to(device)
    print(f"PMI edges: {len(vals)}")

    # Convert to torch sparse
    X_train = scipy_to_torch_sparse(X_train_s).to(device)
    X_val = scipy_to_torch_sparse(X_val_s).to(device)
    X_test = scipy_to_torch_sparse(X_test_s).to(device)
    X_ob = scipy_to_torch_sparse(X_ob_s).to(device)

    y_train = torch.tensor(train_df["fraudulent"].astype(int).values, dtype=torch.long, device=device)
    y_val = torch.tensor(val_df["fraudulent"].astype(int).values, dtype=torch.long, device=device)
    y_test = torch.tensor(test_df["fraudulent"].astype(int).values, dtype=torch.long, device=device)

    # Class weights - SAME AS ORIGINAL
    pos = (y_train == 1).sum().item()
    neg = (y_train == 0).sum().item()
    w0 = 1.0
    w1 = math.sqrt(neg / max(pos, 1))
    class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)
    print(f"Class distribution: neg={neg}, pos={pos}")
    print(f"Class weights: {class_weights.tolist()}")

    # Model
    model = ImprovedWordGCN(
        num_words=num_words,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
        residual_alpha=cfg.residual_alpha,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_f1 = -1.0
    best_state = None
    patience_left = cfg.patience

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        opt.zero_grad()

        logits = model(A_norm, X_train)
        loss = F.cross_entropy(
            logits,
            y_train,
            weight=class_weights,
            label_smoothing=cfg.label_smoothing,
        )
        loss.backward()
        opt.step()

        val_f1, val_p, val_r = eval_split(model, A_norm, X_val, y_val)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | loss={loss.item():.4f} | val_f1={val_f1:.4f} val_p={val_p:.4f} val_r={val_r:.4f}")

        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    # Load best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Find optimal threshold
    best_threshold = find_best_threshold(model, A_norm, X_val, y_val)
    test_f1, test_p, test_r = eval_split(
        model, A_norm, X_test, y_test, threshold=best_threshold["threshold"]
    )

    print(f"\n✅ EMSCAD TEST | F1={test_f1:.4f} Precision={test_p:.4f} Recall={test_r:.4f} "
          f"(threshold={best_threshold['threshold']:.2f})")

    # OpenDataBay evaluation
    model.eval()
    with torch.no_grad():
        ob_logits = model(A_norm, X_ob)
        ob_probs = F.softmax(ob_logits, dim=1)[:, 1].cpu().numpy()
        ob_pred = (ob_probs >= best_threshold["threshold"]).astype(int)
        openbay_recall = float(np.mean(ob_pred == 1))
        print(f"✅ OpenDataBay | recall={openbay_recall:.4f} | "
              f"mean_prob={ob_probs.mean():.4f} median_prob={np.median(ob_probs):.4f}")

    # Save artifacts
    joblib.dump(vec, PATHS.models_textgcn / "vectorizer_improved.joblib")

    torch.save({
        "state_dict": model.state_dict(),
        "num_words": num_words,
        "hidden_dim": cfg.hidden_dim,
        "dropout": cfg.dropout,
        "residual_alpha": cfg.residual_alpha,
    }, PATHS.models_textgcn / "textgcn_improved.pt")

    torch.save({
        "A_norm_indices": A_norm.coalesce().indices().cpu(),
        "A_norm_values": A_norm.coalesce().values().cpu(),
        "A_norm_size": A_norm.shape,
        "inv_vocab": inv_vocab,
    }, PATHS.models_textgcn / "graph_cache_improved.pt")

    metrics = {
        "model": "textgcn_improved",
        "emscad_test_f1": float(test_f1),
        "emscad_test_precision": float(test_p),
        "emscad_test_recall": float(test_r),
        "openbay_recall": float(openbay_recall),
        "openbay_mean_prob": float(ob_probs.mean()),
        "vocab_size": int(num_words),
        "threshold": float(best_threshold["threshold"]),
    }

    pd.DataFrame([metrics]).to_csv(PATHS.reports / "metrics_textgcn_improved.csv", index=False)
    print(f"\n✅ Saved improved model to {PATHS.models_textgcn}/")


if __name__ == "__main__":
    main()