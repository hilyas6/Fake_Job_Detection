from __future__ import annotations

import itertools
import math
import random

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_trial(cfg, device, train_df, val_df, test_df, ob_df):
    set_seed(cfg["seed"])
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
        pmi_threshold=cfg["pmi_threshold"],
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

    model = ImprovedWordGCN(
        n,
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
        residual_alpha=cfg["residual_alpha"],
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="max",
        factor=0.5,
        patience=max(2, cfg["patience"] // 3),
        min_lr=5e-5,
    )

    best_state = None
    best_val = {"f1": -1.0}
    patience = cfg["patience"]
    for _ in range(cfg["epochs"]):
        model.train()
        opt.zero_grad()
        logits = model(A_norm, X_train)
        loss = F.cross_entropy(logits, y_train, weight=class_weights, label_smoothing=cfg["label_smoothing"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            val_probs = F.softmax(model(A_norm, X_val), dim=1)[:, 1].cpu().numpy()

        val = find_best_threshold(y_val, val_probs)
        scheduler.step(val["f1"])

        if val["f1"] > best_val["f1"] + 1e-6:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = cfg["patience"]
        else:
            patience -= 1
            if patience <= 0:
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

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

    common = {
        "weight_decay": 1e-5,
        "patience": 18,
        "epochs": 140,
        "label_smoothing": 0.05,
        "pmi_threshold": 0.0,
        "max_features": 40000,
    }
    grid = itertools.product(
        [41, 42],  # seed
        [(1, 2), (1, 3)],  # ngram_range
        [20],  # window_size
        [300],  # hidden_dim
        [0.3, 0.35],  # dropout
        [2e-3, 3e-3],  # lr
        [0.7],  # residual_alpha
    )
    trials = [
        {
            **common,
            "seed": seed,
            "ngram_range": ngram,
            "window_size": window,
            "hidden_dim": hidden,
            "dropout": dropout,
            "lr": lr,
            "residual_alpha": residual_alpha,
        }
        for seed, ngram, window, hidden, dropout, lr, residual_alpha in grid
    ]

    best = None
    best_f1 = -1.0
    for idx, cfg in enumerate(trials, start=1):
        out = run_trial(cfg, device, bundle.train_df, bundle.val_df, bundle.test_df, bundle.openbay_df)
        val = find_best_threshold(out["y_val"], out["val_probs"])
        print(f"[{idx}/{len(trials)}] val_f1={val['f1']:.4f} thr={val['threshold']:.2f} cfg={cfg}")
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
    print(f"Best val_f1={best_f1:.4f} | threshold={threshold:.2f} | config={best['config']}")


if __name__ == "__main__":
    main()
