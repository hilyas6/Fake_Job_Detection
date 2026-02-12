from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from tuned_models.common import MODELS_DIR, ensure_dirs, eval_at_threshold, find_best_threshold, load_bundle, openbay_metrics, save_metrics_row


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


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    probs = []
    for batch in loader:
        batch.pop("labels", None)
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def run_trial(lr, train_loader, val_loader, y_val, device, epochs=2):
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_f1 = -1.0
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = criterion(model(**batch).logits, labels)
            loss.backward()
            optimizer.step()
        val_probs = predict_probs(model, val_loader, device)
        val_f1 = eval_at_threshold(y_val, val_probs, 0.5)["f1"]
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_f1


def main() -> None:
    ensure_dirs()
    bundle = load_bundle()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=True)
    def tok(texts):
        return tokenizer(texts, truncation=True, max_length=160, padding=False)

    y_train = bundle.train_df["fraudulent"].astype(int).values
    y_val = bundle.val_df["fraudulent"].astype(int).values
    y_test = bundle.test_df["fraudulent"].astype(int).values

    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    train_loader = DataLoader(TokenizedDataset(tok(bundle.train_df["text"].tolist()), y_train), batch_size=12, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(TokenizedDataset(tok(bundle.val_df["text"].tolist()), y_val), batch_size=16, collate_fn=collator)
    test_loader = DataLoader(TokenizedDataset(tok(bundle.test_df["text"].tolist()), y_test), batch_size=16, collate_fn=collator)
    ob_loader = DataLoader(TokenizedDataset(tok(bundle.openbay_df["text"].tolist()), None), batch_size=16, collate_fn=collator)

    best_model = None
    best_f1 = -1.0
    for lr in [1e-5, 2e-5, 3e-5]:
        model, val_f1 = run_trial(lr, train_loader, val_loader, y_val, device)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model = model

    assert best_model is not None
    val_probs = predict_probs(best_model, val_loader, device)
    threshold = find_best_threshold(y_val, val_probs)["threshold"]
    test_probs = predict_probs(best_model, test_loader, device)
    test = eval_at_threshold(y_test, test_probs, threshold)
    ob_probs = predict_probs(best_model, ob_loader, device)

    out_dir = MODELS_DIR / "distilbert"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    save_metrics_row(
        "distilbert",
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
