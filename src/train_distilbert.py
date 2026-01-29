# src/train_distilbert.py
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
    max_len: int = 256
    batch_size: int = 8
    lr: float = 2e-5
    epochs: int = 3
    patience: int = 2


class BertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        if self.labels is None:
            return item
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    probs = []
    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        logits = model(**inputs).logits
        prob = torch.softmax(logits, dim=1)[:, 1]
        probs.append(prob.cpu().numpy())
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

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=2,
    ).to(device)

    train_labels = train_df["fraudulent"].astype(int).values
    val_labels = val_df["fraudulent"].astype(int).values
    test_labels = test_df["fraudulent"].astype(int).values

    train_ds = BertDataset(train_df["text"].tolist(), train_labels, tokenizer, cfg.max_len)
    val_ds = BertDataset(val_df["text"].tolist(), val_labels, tokenizer, cfg.max_len)
    test_ds = BertDataset(test_df["text"].tolist(), test_labels, tokenizer, cfg.max_len)
    ob_ds = BertDataset(ob["text"].tolist(), None, tokenizer, cfg.max_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)
    ob_loader = DataLoader(ob_ds, batch_size=cfg.batch_size)

    pos = float((train_labels == 1).sum())
    neg = float((train_labels == 0).sum())
    class_weights = torch.tensor([1.0, neg / max(pos, 1.0)], device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_state = None
    best_val_f1 = -1.0
    patience_left = cfg.patience

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(**inputs).logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_probs = predict_probs(model, val_loader, device)
        val_f1, val_p, val_r = eval_at_threshold(val_labels, val_probs, 0.5)
        if epoch == 1 or epoch % 1 == 0:
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
    test_f1, test_p, test_r = eval_at_threshold(
        test_labels, test_probs, best_threshold["threshold"]
    )

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
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": cfg.__dict__,
        },
        model_dir / "distilbert.pt",
    )
    pd.DataFrame([metrics]).to_csv(PATHS.reports / "metrics_distilbert.csv", index=False)

    print(f"✅ Saved DistilBERT model to {model_dir}")


if __name__ == "__main__":
    main()
