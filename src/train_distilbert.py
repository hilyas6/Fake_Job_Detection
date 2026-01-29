# src/train_distilbert.py
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from src.config import PATHS
from src.model_utils import (
    eval_at_threshold,
    find_best_threshold,
    load_splits,
    openbay_metrics,
    subset_by_ids,
)


@dataclass
class TrainConfig:
    model_name: str = "distilbert-base-uncased"
    max_len: int = 128          # 256 -> 128 is a big speed boost on CPU/MPS
    batch_size: int = 16        # M1 can often handle 16 at 128 tokens; reduce if OOM
    lr: float = 2e-5
    epochs: int = 3
    patience: int = 2
    use_amp: bool = True        # mixed precision on MPS/CUDA when available


def get_device() -> torch.device:
    # Prefer Apple GPU (MPS) on M1/M2, then CUDA, else CPU
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TokenizedDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item



def tokenize_texts(tokenizer, texts, max_len: int):
    # Return lists (not torch tensors). Dynamic padding will happen in the DataLoader collator.
    return tokenizer(
        texts,
        truncation=True,
        max_length=max_len,
        padding=False,
        return_tensors=None,
    )



@torch.no_grad()
def predict_probs(model, loader, device) -> np.ndarray:
    model.eval()
    probs = []

    for batch in loader:
        labels = batch.pop("labels", None)  # may not exist for OpenBay
        batch = {k: v.to(device) for k, v in batch.items()}

        logits = model(**batch).logits
        prob = torch.softmax(logits, dim=1)[:, 1]
        probs.append(prob.detach().cpu().numpy())

    return np.concatenate(probs, axis=0)


def main():
    PATHS.models_comparison.mkdir(parents=True, exist_ok=True)
    PATHS.reports.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig()
    device = get_device()
    print(f"Using device: {device}")

    # (Optional) helps avoid some MPS edge weirdness in older setups
    # torch.set_float32_matmul_precision("high")  # PyTorch 2.x; safe if available

    em = pd.read_csv(PATHS.data_processed / "emscad.csv")
    ob = pd.read_csv(PATHS.data_processed / "openbay.csv")
    splits = load_splits()

    train_df = subset_by_ids(em, splits["train_ids"])
    val_df = subset_by_ids(em, splits["val_ids"])
    test_df = subset_by_ids(em, splits["test_ids"])

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=2,
    ).to(device)

    train_labels = train_df["fraudulent"].astype(int).values
    val_labels = val_df["fraudulent"].astype(int).values
    test_labels = test_df["fraudulent"].astype(int).values

    # ---- TOKENIZE ONCE (FAST) ----
    train_enc = tokenize_texts(tokenizer, train_df["text"].tolist(), cfg.max_len)
    val_enc   = tokenize_texts(tokenizer, val_df["text"].tolist(), cfg.max_len)
    test_enc  = tokenize_texts(tokenizer, test_df["text"].tolist(), cfg.max_len)
    ob_enc    = tokenize_texts(tokenizer, ob["text"].tolist(), cfg.max_len)

    train_ds = TokenizedDataset(train_enc, train_labels)
    val_ds   = TokenizedDataset(val_enc, val_labels)
    test_ds  = TokenizedDataset(test_enc, test_labels)
    ob_ds    = TokenizedDataset(ob_enc, labels=None)

    # Dynamic padding per batch (MUCH faster than padding every sample to max_len)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    # macOS: keep num_workers=0 to avoid multiprocessing headaches/hangs
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=0, collate_fn=collator
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, collate_fn=collator
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, collate_fn=collator
    )
    ob_loader = DataLoader(
        ob_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, collate_fn=collator
    )

    # Class weights (same as your code)
    pos = float((train_labels == 1).sum())
    neg = float((train_labels == 0).sum())
    class_weights = torch.tensor([1.0, neg / max(pos, 1.0)], device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Mixed precision for MPS/CUDA (can be a big speedup)
    use_amp = cfg.use_amp and device.type in {"cuda", "mps"}
    autocast_dtype = torch.float16  # safest common choice

    best_state = None
    best_val_f1 = -1.0
    patience_left = cfg.patience

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            labels = batch.pop("labels").to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    logits = model(**inputs).logits
                    loss = criterion(logits, labels)
            else:
                logits = model(**inputs).logits
                loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

            # lightweight heartbeat so it never looks "stuck"
            if step == 0:
                print(f"Epoch {epoch:02d} first batch ok. loss={loss.item():.4f}")

        val_probs = predict_probs(model, val_loader, device)
        val_f1, val_p, val_r = eval_at_threshold(val_labels, val_probs, 0.5)

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
    best_threshold = find_best_threshold(val_labels, val_probs)

    test_probs = predict_probs(model, test_loader, device)
    test_f1, test_p, test_r = eval_at_threshold(test_labels, test_probs, best_threshold["threshold"])

    ob_probs = predict_probs(model, ob_loader, device)
    ob_metrics = openbay_metrics(ob_probs, best_threshold["threshold"])

    metrics = {
        "model": "distilbert",
        "emscad_test_f1": float(test_f1),
        "emscad_test_precision": float(test_p),
        "emscad_test_recall": float(test_r),
        "openbay_recall": ob_metrics["openbay_recall"],
        "openbay_mean_prob": ob_metrics["openbay_mean_prob"],
        "openbay_median_prob": ob_metrics["openbay_median_prob"],
        "threshold": float(best_threshold["threshold"]),
    }

    model_dir = PATHS.models_comparison / "distilbert"
    model_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    torch.save({"state_dict": model.state_dict(), "config": cfg.__dict__}, model_dir / "distilbert.pt")
    pd.DataFrame([metrics]).to_csv(PATHS.reports / "metrics_distilbert.csv", index=False)

    print(f"✅ Saved DistilBERT model to {model_dir}")


if __name__ == "__main__":
    main()
