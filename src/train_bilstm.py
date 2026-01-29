# src/train_bilstm.py
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.config import PATHS
from src.model_utils import (
    encode_tokens,
    eval_at_threshold,
    find_best_threshold,
    load_splits,
    openbay_metrics,
    subset_by_ids,
    tokenize,
    build_vocab,
)


@dataclass
class TrainConfig:
    max_len: int = 400
    max_vocab: int = 40000
    min_freq: int = 2
    embed_dim: int = 200
    hidden_dim: int = 128
    dropout: float = 0.3
    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 12
    patience: int = 3


class TextDataset(Dataset):
    def __init__(self, sequences, labels=None, pad_idx=0, max_len=400):
        self.sequences = sequences
        self.labels = labels
        self.pad_idx = pad_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def _pad(self, seq):
        if len(seq) >= self.max_len:
            return seq[: self.max_len]
        return seq + [self.pad_idx] * (self.max_len - len(seq))

    def __getitem__(self, idx):
        seq = self._pad(self.sequences[idx])
        if self.labels is None:
            return torch.tensor(seq, dtype=torch.long)
        return torch.tensor(seq, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float32)


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        emb = self.embedding(x)
        out, (h_n, _) = self.lstm(emb)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        h = self.dropout(h)
        return self.fc(h).squeeze(1)


def build_sequences(texts, vocab):
    sequences = []
    for text in texts:
        tokens = tokenize(text)
        sequences.append(encode_tokens(tokens, vocab))
    return sequences


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    probs = []
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        batch = batch.to(device)
        logits = model(batch)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs, axis=0)


def main():
    PATHS.models_comparison.mkdir(parents=True, exist_ok=True)
    PATHS.reports.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    em = pd.read_csv(PATHS.data_processed / "emscad.csv")
    ob = pd.read_csv(PATHS.data_processed / "openbay.csv")
    splits = load_splits()

    train_df = subset_by_ids(em, splits["train_ids"])
    val_df = subset_by_ids(em, splits["val_ids"])
    test_df = subset_by_ids(em, splits["test_ids"])

    train_tokens = [tokenize(t) for t in train_df["text"].tolist()]
    vocab = build_vocab(train_tokens, min_freq=cfg.min_freq, max_size=cfg.max_vocab)

    train_seq = [encode_tokens(tokens, vocab) for tokens in train_tokens]
    val_seq = build_sequences(val_df["text"].tolist(), vocab)
    test_seq = build_sequences(test_df["text"].tolist(), vocab)
    ob_seq = build_sequences(ob["text"].tolist(), vocab)

    y_train = train_df["fraudulent"].astype(int).values
    y_val = val_df["fraudulent"].astype(int).values
    y_test = test_df["fraudulent"].astype(int).values

    train_ds = TextDataset(train_seq, y_train, max_len=cfg.max_len)
    val_ds = TextDataset(val_seq, y_val, max_len=cfg.max_len)
    test_ds = TextDataset(test_seq, y_test, max_len=cfg.max_len)
    ob_ds = TextDataset(ob_seq, labels=None, max_len=cfg.max_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)
    ob_loader = DataLoader(ob_ds, batch_size=cfg.batch_size)

    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
    ).to(device)

    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_state = None
    best_val_f1 = -1.0
    patience_left = cfg.patience

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_probs = predict_probs(model, val_loader, device)
        val_f1, val_p, val_r = eval_at_threshold(y_val, val_probs, 0.5)
        if epoch % 2 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:02d} | loss={total_loss/len(train_loader):.4f} | "
                f"val_f1={val_f1:.4f} val_p={val_p:.4f} val_r={val_r:.4f}"
            )

        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_probs = predict_probs(model, val_loader, device)
    best_threshold = find_best_threshold(y_val, val_probs)

    test_probs = predict_probs(model, test_loader, device)
    test_f1, test_p, test_r = eval_at_threshold(
        y_test, test_probs, best_threshold["threshold"]
    )

    ob_probs = predict_probs(model, ob_loader, device)
    ob_metrics = openbay_metrics(ob_probs, best_threshold["threshold"])

    metrics = {
        "model": "bilstm",
        "emscad_test_f1": float(test_f1),
        "emscad_test_precision": float(test_p),
        "emscad_test_recall": float(test_r),
        "openbay_recall": ob_metrics["openbay_recall"],
        "openbay_mean_prob": ob_metrics["openbay_mean_prob"],
        "openbay_median_prob": ob_metrics["openbay_median_prob"],
        "threshold": float(best_threshold["threshold"]),
    }

    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": vocab,
            "config": cfg.__dict__,
        },
        PATHS.models_comparison / "bilstm.pt",
    )
    pd.DataFrame([metrics]).to_csv(PATHS.reports / "metrics_bilstm.csv", index=False)

    print(f"✅ Saved Bi-LSTM model to {PATHS.models_comparison}/bilstm.pt")


if __name__ == "__main__":
    main()
