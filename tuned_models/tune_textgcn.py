from __future__ import annotations

import math

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer

from tuned_models.common import MODELS_DIR, ensure_dirs, eval_at_threshold, find_best_threshold, load_bundle, openbay_metrics, save_metrics_row
from src.train_textgcn_enhanced import (
    ImprovedWordGCN,
    build_pmi_graph,
    normalize_sparse_adj,
    scipy_to_torch_sparse,
    tokenize,
)


def run_trial(cfg, device, train_df, val_df, test_df, ob_df):
    vec = TfidfVectorizer(
        tokenizer=tokenize,
        preprocessor=None,
        token_pattern=None,
        ngram_range=cfg["ngram_range"],
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        max_features=cfg["max_features"],
    )
    X_train_s = vec.fit_transform(train_df["text"])
    X_val_s = vec.transform(val_df["text"])
    X_test_s = vec.transform(test_df["text"])
    X_ob_s = vec.transform(ob_df["text"])

    rows, cols, vals, n = build_pmi_graph(
        [tokenize(t) for t in train_df["text"].tolist()],
        vec.vocabulary_,
        window_size=cfg["window_size"],
        pmi_threshold=0.0,
    )
    A_norm = normalize_sparse_adj(rows, cols, vals, n).to(device)
    X_train = scipy_to_torch_sparse(X_train_s).to(device)
    X_val = scipy_to_torch_sparse(X_val_s).to(device)
    X_test = scipy_to_torch_sparse(X_test_s).to(device)
    X_ob = scipy_to_torch_sparse(X_ob_s).to(device)

    y_train = torch.tensor(train_df["fraudulent"].astype(int).values, dtype=torch.long, device=device)
    y_val = val_df["fraudulent"].astype(int).values
    y_test = test_df["fraudulent"].astype(int).values

    pos = int((y_train == 1).sum().item())
    neg = int((y_train == 0).sum().item())
    class_weights = torch.tensor([1.0, math.sqrt(neg / max(pos, 1))], dtype=torch.float32, device=device)

    model = ImprovedWordGCN(n, hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"], residual_alpha=0.7).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)

    best_state = None
    best_f1 = -1.0
    patience = 3
    for _ in range(cfg["epochs"]):
        model.train()
        opt.zero_grad()
        logits = model(A_norm, X_train)
        loss = F.cross_entropy(logits, y_train, weight=class_weights, label_smoothing=0.05)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_probs = F.softmax(model(A_norm, X_val), dim=1)[:, 1].cpu().numpy()
        val_f1 = eval_at_threshold(y_val, val_probs, 0.5)["f1"]
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 3
        else:
            patience -= 1
            if patience <= 0:
                break

    model.load_state_dict(best_state)
    with torch.no_grad():
        val_probs = F.softmax(model(A_norm, X_val), dim=1)[:, 1].cpu().numpy()
        test_probs = F.softmax(model(A_norm, X_test), dim=1)[:, 1].cpu().numpy()
        ob_probs = F.softmax(model(A_norm, X_ob), dim=1)[:, 1].cpu().numpy()

    return {
        "model": model,
        "vectorizer": vec,
        "A_norm": A_norm,
        "val_probs": val_probs,
        "test_probs": test_probs,
        "ob_probs": ob_probs,
        "y_val": y_val,
        "y_test": y_test,
        "config": cfg,
    }


def main() -> None:
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_bundle()

    trials = [
        {"ngram_range": (1, 2), "max_features": 40000, "window_size": 20, "hidden_dim": 256, "dropout": 0.35, "lr": 3e-3, "epochs": 40},
        {"ngram_range": (1, 3), "max_features": 50000, "window_size": 22, "hidden_dim": 300, "dropout": 0.3, "lr": 2e-3, "epochs": 50},
    ]

    best = None
    best_f1 = -1.0
    for cfg in trials:
        out = run_trial(cfg, device, bundle.train_df, bundle.val_df, bundle.test_df, bundle.openbay_df)
        val = find_best_threshold(out["y_val"], out["val_probs"])
        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            best = {**out, "val_best": val}

    threshold = best["val_best"]["threshold"]
    test = eval_at_threshold(best["y_test"], best["test_probs"], threshold)

    torch.save({"state_dict": best["model"].state_dict(), "config": best["config"]}, MODELS_DIR / "textgcn.pt")
    joblib.dump(best["vectorizer"], MODELS_DIR / "textgcn_vectorizer.joblib")

    save_metrics_row(
        "textgcn",
        {
            "emscad_test_f1": test["f1"],
            "emscad_test_precision": test["precision"],
            "emscad_test_recall": test["recall"],
            "threshold": threshold,
            **openbay_metrics(best["ob_probs"], threshold),
        },
    )


if __name__ == "__main__":
    main()
