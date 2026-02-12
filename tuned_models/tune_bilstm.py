from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

from tuned_models.common import (
    MODELS_DIR,
    ensure_dirs,
    eval_at_threshold,
    find_best_threshold,
    load_bundle,
    openbay_metrics,
    save_metrics_row,
)
from src.model_utils import build_vocab, encode_tokens, tokenize


class TextDataset(Dataset):
    def __init__(self, seqs, labels=None, max_len=400):
        self.seqs = seqs
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx][: self.max_len]
        length = max(1, len(seq))
        if len(seq) < self.max_len:
            seq = seq + [0] * (self.max_len - len(seq))
        x = torch.tensor(seq, dtype=torch.long)
        if self.labels is None:
            return x, torch.tensor(length, dtype=torch.long)
        return x, torch.tensor(length, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float32)


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(min(0.25, dropout))
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, lengths):
        emb = self.embedding_dropout(self.embedding(x))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc(self.dropout(h)).squeeze(1)


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    out = []
    for batch in loader:
        if isinstance(batch, (tuple, list)):
            x, lengths = batch[:2]
        else:
            x = batch
            lengths = torch.full((x.shape[0],), x.shape[1], dtype=torch.long)
        logits = model(x.to(device), lengths.to(device))
        out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)


def build_sequences(texts, vocab):
    return [encode_tokens(tokenize(text), vocab) for text in texts]


def fit_and_eval(cfg, seq_data, y_data, loaders, device):
    model = BiLSTMClassifier(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
        num_layers=cfg["num_layers"],
    ).to(device)
    pos = float((y_data["train"] == 1).sum())
    neg = float((y_data["train"] == 0).sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
    )

    best_f1 = -1.0
    best_state = None
    patience = 3
    for _ in range(cfg["epochs"]):
        model.train()
        for x, lengths, y in loaders["train"]:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x, lengths), y)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=cfg["grad_clip"])
            optimizer.step()
        val_probs = predict_probs(model, loaders["val"], device)
        val_f1 = find_best_threshold(y_data["val"], val_probs)["f1"]
        scheduler.step(val_f1)
        if val_f1 > best_f1 + 1e-4:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 3
        else:
            patience -= 1
            if patience <= 0:
                break

    model.load_state_dict(best_state)
    return model, best_f1


def main() -> None:
    ensure_dirs()
    torch.manual_seed(42)
    np.random.seed(42)
    bundle = load_bundle()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tokens = [tokenize(t) for t in bundle.train_df["text"].tolist()]
    vocab = build_vocab(train_tokens, min_freq=2, max_size=50000)

    seqs = {
        "train": [encode_tokens(tokens, vocab) for tokens in train_tokens],
        "val": build_sequences(bundle.val_df["text"].tolist(), vocab),
        "test": build_sequences(bundle.test_df["text"].tolist(), vocab),
        "ob": build_sequences(bundle.openbay_df["text"].tolist(), vocab),
    }
    ys = {
        "train": bundle.train_df["fraudulent"].astype(int).values,
        "val": bundle.val_df["fraudulent"].astype(int).values,
        "test": bundle.test_df["fraudulent"].astype(int).values,
    }

    trials = [
        {
            "embed_dim": 200,
            "hidden_dim": 128,
            "dropout": 0.25,
            "lr": 8e-4,
            "epochs": 10,
            "num_layers": 2,
            "weight_decay": 1e-4,
            "grad_clip": 1.0,
            "max_len": 360,
        },
        {
            "embed_dim": 256,
            "hidden_dim": 160,
            "dropout": 0.3,
            "lr": 1e-3,
            "epochs": 12,
            "num_layers": 2,
            "weight_decay": 1e-4,
            "grad_clip": 1.0,
            "max_len": 420,
        },
        {
            "embed_dim": 256,
            "hidden_dim": 192,
            "dropout": 0.35,
            "lr": 7e-4,
            "epochs": 12,
            "num_layers": 2,
            "weight_decay": 2e-4,
            "grad_clip": 0.8,
            "max_len": 480,
        },
    ]

    best_trial = None
    best_model = None
    best_f1 = -1.0
    for trial in trials:
        max_len = trial["max_len"]
        cfg = {**trial, "vocab_size": len(vocab)}
        loaders = {
            "train": DataLoader(TextDataset(seqs["train"], ys["train"], max_len), batch_size=32, shuffle=True),
            "val": DataLoader(TextDataset(seqs["val"], ys["val"], max_len), batch_size=64),
            "test": DataLoader(TextDataset(seqs["test"], ys["test"], max_len), batch_size=64),
            "ob": DataLoader(TextDataset(seqs["ob"], None, max_len), batch_size=64),
        }
        model, val_f1 = fit_and_eval(cfg, seqs, ys, loaders, device)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model = model
            best_trial = cfg
            best_loaders = loaders

    val_probs = predict_probs(best_model, best_loaders["val"], device)
    threshold = find_best_threshold(ys["val"], val_probs)["threshold"]
    test_probs = predict_probs(best_model, best_loaders["test"], device)
    test = eval_at_threshold(ys["test"], test_probs, threshold)
    ob_probs = predict_probs(best_model, best_loaders["ob"], device)

    torch.save(
        {"state_dict": best_model.state_dict(), "vocab": vocab, "config": best_trial},
        MODELS_DIR / "bilstm.pt",
    )

    save_metrics_row(
        "bilstm",
        {
            "emscad_test_f1": test["f1"],
            "emscad_test_precision": test["precision"],
            "emscad_test_recall": test["recall"],
            "threshold": threshold,
            **openbay_metrics(ob_probs, threshold),
        },
    )


if __name__ == "__main__":
    main()
