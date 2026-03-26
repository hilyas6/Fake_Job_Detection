"""
3-Fold Stratified Cross-Validation
=====================================
Evaluates TextGCN (single-seed) and LightGBM on all available EMSCAD data
using 3-fold stratified CV.  Reports mean ± std F1 to demonstrate that
performance differences are consistent, not split-dependent.

Why only these two?
  - TextGCN: the novel contribution — needs rigorous validation
  - LightGBM: the strongest classical baseline

DistilBERT is excluded due to prohibitive training time per fold.

Outputs
-------
  reports/cross_validation_results.csv
  reports/cross_validation_summary.txt
"""
import json
import math
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS_DIR    = ROOT / "reports"

from tuned_models.tune_textgcn import (
    ImprovedWordGCN,
    build_pmi_graph, normalize_adj, to_sparse, tokenize,
    best_threshold, eval_at, get_probs,
)


# ── LightGBM fold ─────────────────────────────────────────────────────────────
def run_lgbm_fold(X_tr_s, y_tr, X_va_s, y_va, X_te_s, y_te):
    from lightgbm import LGBMClassifier
    neg = int((y_tr == 0).sum())
    pos = int((y_tr == 1).sum())
    clf = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, subsample=0.8,
        scale_pos_weight=neg / max(pos, 1),
        n_jobs=-1, random_state=42, verbose=-1,
    )
    clf.fit(X_tr_s, y_tr)
    va_probs = clf.predict_proba(X_va_s)[:, 1]
    te_probs = clf.predict_proba(X_te_s)[:, 1]
    thr = best_threshold(va_probs, y_va)
    f1, p, r = eval_at(te_probs, y_te, thr["t"])
    return f1, p, r, thr["t"]


# ── TextGCN fold ──────────────────────────────────────────────────────────────
def run_textgcn_fold(train_texts, val_texts, test_texts, y_tr, y_va, y_te,
                     device, seed=42):
    vec = TfidfVectorizer(
        tokenizer=tokenize, preprocessor=None, token_pattern=None,
        ngram_range=(1, 3), min_df=2, max_df=0.9,
        sublinear_tf=True, max_features=40_000,
    )
    X_tr_s = vec.fit_transform(train_texts)
    X_va_s = vec.transform(val_texts)
    X_te_s = vec.transform(test_texts)

    vocab     = vec.vocabulary_
    num_words = len(vocab)

    tok_train = [tokenize(t) for t in train_texts]
    rows, cols, vals, n = build_pmi_graph(tok_train, vocab, window_size=15)
    A = normalize_adj(rows, cols, vals, n).to(device)

    X_tr = to_sparse(X_tr_s).to(device)
    X_va = to_sparse(X_va_s).to(device)
    X_te = to_sparse(X_te_s).to(device)

    pos = int((y_tr == 1).sum())
    neg = int((y_tr == 0).sum())
    cw  = torch.tensor([1.0, math.sqrt(neg / max(pos, 1))],
                       dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=device)

    torch.manual_seed(seed)
    model = ImprovedWordGCN(num_words, 300, 0.35, 0.7).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)

    best_state, best_vf1, patience_left = None, -1.0, 20

    for ep in range(1, 201):
        model.train()
        opt.zero_grad()
        F.cross_entropy(model(A, X_tr), y_tr_t, weight=cw,
                        label_smoothing=0.05).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        vp = get_probs(model, A, X_va)
        vf1, _, _ = eval_at(vp, y_va)
        if vf1 > best_vf1 + 1e-4:
            best_vf1     = vf1
            best_state   = {k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()}
            patience_left = 20
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state:
        model.load_state_dict(best_state)

    val_probs  = get_probs(model, A, X_va)
    test_probs = get_probs(model, A, X_te)
    thr        = best_threshold(val_probs, y_va)
    f1, p, r   = eval_at(test_probs, y_te, thr["t"])
    return f1, p, r, thr["t"]


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    print("Loading EMSCAD data (all splits combined for CV)...")
    em = pd.read_csv(DATA_PROCESSED / "emscad.csv")
    texts  = em["text"].tolist()
    labels = em["fraudulent"].astype(int).values

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    folds = list(skf.split(texts, labels))
    print(f"3-fold CV | Total samples: {len(em)} | "
          f"Fake: {labels.sum()} ({labels.mean()*100:.1f}%)")

    results = []

    # ── TextGCN ────────────────────────────────────────────────────────────────
    print("\n--- TextGCN (single-seed, window=15) ---")
    for fold_i, (tr_idx, te_idx) in enumerate(folds, 1):
        # Use 80% of fold-train for training, 20% for validation (threshold tuning)
        val_split   = int(0.2 * len(tr_idx))
        va_idx      = tr_idx[:val_split]
        actual_tr_idx = tr_idx[val_split:]

        tr_texts = [texts[i] for i in actual_tr_idx]
        va_texts = [texts[i] for i in va_idx]
        te_texts = [texts[i] for i in te_idx]
        y_tr = labels[actual_tr_idx]
        y_va = labels[va_idx]
        y_te = labels[te_idx]

        print(f"  Fold {fold_i}: train={len(tr_texts)} val={len(va_texts)} "
              f"test={len(te_texts)} (fake in test: {y_te.sum()})")
        t0 = time.time()
        f1, p, r, thr = run_textgcn_fold(
            tr_texts, va_texts, te_texts, y_tr, y_va, y_te, device
        )
        elapsed = time.time() - t0
        print(f"    F1={f1:.4f}  P={p:.4f}  R={r:.4f}  thr={thr:.2f}  ({elapsed:.0f}s)")
        results.append({"model": "TextGCN", "fold": fold_i, "f1": f1,
                        "precision": p, "recall": r, "threshold": thr})

    # ── LightGBM ──────────────────────────────────────────────────────────────
    print("\n--- LightGBM ---")
    vec_lgbm = TfidfVectorizer(
        ngram_range=(1, 3), min_df=2, max_df=0.9,
        sublinear_tf=True, max_features=40_000,
    )
    X_all = vec_lgbm.fit_transform(texts)

    for fold_i, (tr_idx, te_idx) in enumerate(folds, 1):
        val_split     = int(0.2 * len(tr_idx))
        va_idx        = tr_idx[:val_split]
        actual_tr_idx = tr_idx[val_split:]

        y_tr = labels[actual_tr_idx]
        y_va = labels[va_idx]
        y_te = labels[te_idx]

        print(f"  Fold {fold_i}: train={len(actual_tr_idx)} "
              f"val={len(va_idx)} test={len(te_idx)}")
        t0 = time.time()
        f1, p, r, thr = run_lgbm_fold(
            X_all[actual_tr_idx], y_tr,
            X_all[va_idx],        y_va,
            X_all[te_idx],        y_te,
        )
        elapsed = time.time() - t0
        print(f"    F1={f1:.4f}  P={p:.4f}  R={r:.4f}  thr={thr:.2f}  ({elapsed:.0f}s)")
        results.append({"model": "LightGBM", "fold": fold_i, "f1": f1,
                        "precision": p, "recall": r, "threshold": thr})

    df = pd.DataFrame(results)
    df.to_csv(REPORTS_DIR / "cross_validation_results.csv", index=False)

    # Summary with mean ± std
    lines = [
        "3-FOLD CROSS-VALIDATION RESULTS",
        "=" * 55,
        f"{'Model':<12} {'Fold 1':>8} {'Fold 2':>8} {'Fold 3':>8} "
        f"{'Mean':>8} {'±Std':>8}",
        "-" * 55,
    ]
    for model_name in ["TextGCN", "LightGBM"]:
        sub   = df[df["model"] == model_name].sort_values("fold")
        f1s   = sub["f1"].values
        folds_str = "  ".join(f"{v:.4f}" for v in f1s)
        lines.append(
            f"{model_name:<12} {f1s[0]:>8.4f} {f1s[1]:>8.4f} {f1s[2]:>8.4f} "
            f"{f1s.mean():>8.4f} {f1s.std():>8.4f}"
        )

    lines.append("=" * 55)
    lines.append(
        "\nNote: TextGCN uses a single seed per fold (not the 3-seed ensemble)"
        "\nso absolute values are slightly lower than the reported tuned model."
    )

    # Statistical note
    textgcn_f1s = df[df["model"] == "TextGCN"]["f1"].values
    lgbm_f1s    = df[df["model"] == "LightGBM"]["f1"].values
    lines.append(f"\nTextGCN: {textgcn_f1s.mean():.4f} ± {textgcn_f1s.std():.4f}")
    lines.append(f"LightGBM: {lgbm_f1s.mean():.4f} ± {lgbm_f1s.std():.4f}")
    if textgcn_f1s.mean() > lgbm_f1s.mean():
        lines.append("TextGCN leads across CV folds.")
    else:
        diff = lgbm_f1s.mean() - textgcn_f1s.mean()
        lines.append(f"LightGBM leads by {diff:.4f} on average across CV folds.")

    summary = "\n".join(lines)
    print("\n" + summary)

    summary_path = REPORTS_DIR / "cross_validation_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"\nSaved: {REPORTS_DIR / 'cross_validation_results.csv'}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
