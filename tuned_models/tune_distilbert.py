from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from tuned_models.common import MODELS_DIR, ensure_dirs, eval_at_threshold, find_best_threshold, load_bundle, openbay_metrics, save_metrics_row

MAX_LEN      = 256   # was 160, extended to capture more of long job descriptions
EPOCHS       = 4     # was 2, transformers need at least 3 epochs to fine-tune properly
PATIENCE     = 2     # early stopping on val F1
BATCH_TRAIN  = 12
BATCH_EVAL   = 16
WARMUP_FRAC  = 0.10  # first 10% of steps = linear warmup
GRAD_CLIP    = 1.0


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


def make_scheduler(optimizer, total_steps: int):
    """Linear warmup for first WARMUP_FRAC of steps, then linear decay to 0."""
    warmup_steps = max(1, int(total_steps * WARMUP_FRAC))
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    decay  = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps - warmup_steps)
    return SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[warmup_steps])


def run_trial(lr: float, train_loader, val_loader, y_val, class_weights, device):
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = EPOCHS * len(train_loader)
    scheduler = make_scheduler(optimizer, total_steps)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_state, best_f1, no_improve = None, -1.0, 0

    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch  = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = criterion(model(**batch).logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

        val_probs = predict_probs(model, val_loader, device)
        val_f1    = eval_at_threshold(y_val, val_probs, 0.5)["f1"]
        print(f"  lr={lr:.0e}  epoch={epoch+1}/{EPOCHS}  val_f1={val_f1:.4f}")

        if val_f1 > best_f1 + 1e-4:
            best_f1     = val_f1
            best_state  = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model, best_f1


def main() -> None:
    ensure_dirs()
    bundle = load_bundle()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=True)
    def tok(texts):
        return tokenizer(texts, truncation=True, max_length=MAX_LEN, padding=False)

    y_train = bundle.train_df["fraudulent"].astype(int).values
    y_val   = bundle.val_df["fraudulent"].astype(int).values
    y_test  = bundle.test_df["fraudulent"].astype(int).values

    # Class weights: weight fake class by sqrt(neg/pos) to handle ~22:1 imbalance
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    class_weights = torch.tensor([1.0, float(np.sqrt(n_neg / n_pos))], dtype=torch.float32)
    print(f"Class weights: real={class_weights[0]:.3f}, fake={class_weights[1]:.3f}")

    collator     = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    train_loader = DataLoader(TokenizedDataset(tok(bundle.train_df["text"].tolist()), y_train),
                              batch_size=BATCH_TRAIN, shuffle=True, collate_fn=collator)
    val_loader   = DataLoader(TokenizedDataset(tok(bundle.val_df["text"].tolist()),   y_val),
                              batch_size=BATCH_EVAL, collate_fn=collator)
    test_loader  = DataLoader(TokenizedDataset(tok(bundle.test_df["text"].tolist()),  y_test),
                              batch_size=BATCH_EVAL, collate_fn=collator)
    ob_loader    = DataLoader(TokenizedDataset(tok(bundle.openbay_df["text"].tolist()), None),
                              batch_size=BATCH_EVAL, collate_fn=collator)

    best_model, best_f1 = None, -1.0
    for lr in [1e-5, 2e-5, 3e-5]:
        print(f"\nTrial lr={lr:.0e}")
        model, val_f1 = run_trial(lr, train_loader, val_loader, y_val, class_weights, device)
        if val_f1 > best_f1:
            best_f1    = val_f1
            best_model = model

    assert best_model is not None
    print(f"\nBest val F1: {best_f1:.4f} - running test evaluation")

    val_probs  = predict_probs(best_model, val_loader, device)
    threshold  = find_best_threshold(y_val, val_probs)["threshold"]
    test_probs = predict_probs(best_model, test_loader, device)
    test       = eval_at_threshold(y_test, test_probs, threshold)
    ob_probs   = predict_probs(best_model, ob_loader, device)

    print(f"Test: F1={test['f1']:.4f}  P={test['precision']:.4f}  R={test['recall']:.4f}  threshold={threshold:.2f}")
    print(f"OpenBay recall: {float(np.mean((ob_probs >= threshold).astype(int))):.4f}")

    out_dir = MODELS_DIR / "distilbert"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    save_metrics_row("distilbert", {
        "emscad_test_f1":        test["f1"],
        "emscad_test_precision": test["precision"],
        "emscad_test_recall":    test["recall"],
        "threshold":             threshold,
        **openbay_metrics(ob_probs, threshold),
    })
    print("Done. Artifacts saved to", out_dir)


if __name__ == "__main__":
    main()
