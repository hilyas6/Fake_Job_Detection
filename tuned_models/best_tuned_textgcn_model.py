"""
Best Tuned TextGCN — Training Script
======================================
Trains the winning configuration (3-seed ensemble + PMI window=15) and saves
model artifacts and metrics ready for comparison.

Run:
    python tuned_models/best_tuned_textgcn_model.py

Outputs:
    models/tuned/textgcn_tuned/textgcn_tuned.pt       — model weights (seed 42)
    models/tuned/textgcn_tuned/graph_cache_tuned.pt   — PMI graph + vocabulary
    models/tuned/textgcn_tuned/vectorizer_tuned.joblib — fitted TF-IDF vectorizer
    reports/tuned/metrics_textgcn_tuned.csv            — metrics for comparison
"""
import math
import sys
from pathlib import Path

import joblib
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tuned_models.tune_textgcn import (
    TrialConfig,
    ImprovedWordGCN,
    load_data,
    build_pmi_graph,
    normalize_adj,
    to_sparse,
    tokenize,
    train_single,
    best_threshold,
    eval_at,
    get_probs,
    TUNED_MODEL_DIR,
    REPORTS_DIR,
)

# ── Winning configuration ───────────────────────────────────────────────────────
CFG = TrialConfig(
    name            = "ensemble_3seeds_window15",
    hidden_dim      = 300,
    dropout         = 0.35,
    residual_alpha  = 0.7,
    lr              = 3e-3,
    weight_decay    = 1e-5,
    optimizer       = "adam",
    scheduler       = "none",
    loss            = "ce",
    epochs          = 200,
    patience        = 25,
    label_smoothing = 0.05,
    class_weight    = "sqrt",
    window_size     = 15,       # KEY CHANGE from baseline (was 20)
    max_features    = 40_000,
    n_seeds         = 3,        # ensemble of 3 independently initialised models
    base_seed       = 42,       # seeds used: 42, 43, 44
)

BASELINE = {"f1": 0.8352, "precision": 0.8769, "recall": 0.7972, "ob_recall": 0.9583}


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TUNED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ───────────────────────────────────────────────────────────────
    bundle = load_data()
    train_df, val_df, test_df, ob_df = (
        bundle["train"], bundle["val"], bundle["test"], bundle["ob"]
    )
    print(f"Train: {len(train_df)}  Val: {len(val_df)}"
          f"  Test: {len(test_df)}  OpenBay: {len(ob_df)}")

    # ── TF-IDF vectorizer ───────────────────────────────────────────────────────
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(
        tokenizer=tokenize, preprocessor=None, token_pattern=None,
        ngram_range=(1, 3), min_df=2, max_df=0.9,
        sublinear_tf=True, max_features=CFG.max_features,
    )
    X_tr_s = vec.fit_transform(train_df["text"])
    X_va_s = vec.transform(val_df["text"])
    X_te_s = vec.transform(test_df["text"])
    X_ob_s = vec.transform(ob_df["text"])

    vocab     = vec.vocabulary_
    num_words = len(vocab)
    print(f"\nVocab size : {num_words}")
    print(f"PMI window : {CFG.window_size}")

    # ── PMI graph ───────────────────────────────────────────────────────────────
    tok_train = [tokenize(t) for t in train_df["text"].tolist()]
    rows, cols, vals, n = build_pmi_graph(tok_train, vocab, window_size=CFG.window_size)
    A = normalize_adj(rows, cols, vals, n).to(device)
    print(f"PMI edges  : {len(vals)}")

    X_tr = to_sparse(X_tr_s).to(device)
    X_va = to_sparse(X_va_s).to(device)
    X_te = to_sparse(X_te_s).to(device)
    X_ob = to_sparse(X_ob_s).to(device)

    y_tr = torch.tensor(train_df["fraudulent"].astype(int).values, dtype=torch.long, device=device)
    y_va = torch.tensor(val_df["fraudulent"].astype(int).values,   dtype=torch.long, device=device)
    y_te = torch.tensor(test_df["fraudulent"].astype(int).values,  dtype=torch.long, device=device)
    y_va_np = y_va.cpu().numpy()
    y_te_np = y_te.cpu().numpy()

    # ── Train 3-seed ensemble ───────────────────────────────────────────────────
    print(f"\nTraining {CFG.n_seeds}-seed ensemble (seeds {CFG.base_seed}–{CFG.base_seed + CFG.n_seeds - 1})")
    all_val, all_test, all_ob = [], [], []
    for s in range(CFG.n_seeds):
        seed = CFG.base_seed + s
        print(f"  → seed {seed}")
        vp, tp, op, best_vf1 = train_single(
            CFG, A, X_tr, X_va, X_te, X_ob, y_tr, y_va, num_words, device, seed_offset=s
        )
        all_val.append(vp)
        all_test.append(tp)
        all_ob.append(op)
        print(f"    best val F1 = {best_vf1:.4f}")

    val_probs  = np.mean(all_val,  axis=0)
    test_probs = np.mean(all_test, axis=0)
    ob_probs   = np.mean(all_ob,   axis=0)

    # ── Threshold tuning and evaluation ────────────────────────────────────────
    thr = best_threshold(val_probs, y_va_np)
    test_f1, test_p, test_r = eval_at(test_probs, y_te_np, thr["t"])
    ob_recall = float(np.mean((ob_probs >= thr["t"]).astype(int) == 1))

    # ── Print results ───────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"{'Metric':<22} {'Baseline':>10} {'Tuned':>10} {'Δ':>8}")
    print(f"{'─'*50}")
    print(f"{'EMSCAD F1':<22} {BASELINE['f1']:>10.4f} {test_f1:>10.4f} {test_f1 - BASELINE['f1']:>+8.4f}")
    print(f"{'EMSCAD Precision':<22} {BASELINE['precision']:>10.4f} {test_p:>10.4f} {test_p - BASELINE['precision']:>+8.4f}")
    print(f"{'EMSCAD Recall':<22} {BASELINE['recall']:>10.4f} {test_r:>10.4f} {test_r - BASELINE['recall']:>+8.4f}")
    print(f"{'OpenBay Recall':<22} {BASELINE['ob_recall']:>10.4f} {ob_recall:>10.4f} {ob_recall - BASELINE['ob_recall']:>+8.4f}")
    print(f"{'Threshold':<22} {'0.48':>10} {thr['t']:>10.2f}")
    print(f"{'─'*50}")

    beats = test_f1 > BASELINE["f1"]
    print(f"\n{'✅ Beats baseline' if beats else '❌ Below baseline'} (baseline F1 = {BASELINE['f1']:.4f})")

    # ── Save metrics ────────────────────────────────────────────────────────────
    pd.DataFrame([{
        "model":                 "textgcn_tuned",
        "emscad_test_f1":        round(test_f1, 6),
        "emscad_test_precision": round(test_p,  6),
        "emscad_test_recall":    round(test_r,  6),
        "openbay_recall":        round(ob_recall, 4),
        "vocab_size":            num_words,
        "threshold":             round(thr["t"], 2),
        "trial":                 CFG.name,
    }]).to_csv(REPORTS_DIR / "metrics_textgcn_tuned.csv", index=False)

    # ── Save graph + vectorizer ─────────────────────────────────────────────────
    inv_vocab = {i: t for t, i in vocab.items()}
    joblib.dump(vec, TUNED_MODEL_DIR / "vectorizer_tuned.joblib")
    torch.save({
        "A_norm_indices": A.coalesce().indices().cpu(),
        "A_norm_values":  A.coalesce().values().cpu(),
        "A_norm_size":    A.shape,
        "inv_vocab":      inv_vocab,
    }, TUNED_MODEL_DIR / "graph_cache_tuned.pt")

    # ── Re-train and save seed-42 model for single-model serving ───────────────
    print("\nSaving seed-42 model for inference...")
    torch.manual_seed(CFG.base_seed)
    save_m = ImprovedWordGCN(num_words, CFG.hidden_dim, CFG.dropout, CFG.residual_alpha).to(device)
    pos_n = int((y_tr == 1).sum()); neg_n = int((y_tr == 0).sum())
    cw = torch.tensor([1.0, math.sqrt(neg_n / max(pos_n, 1))], dtype=torch.float32, device=device)
    save_opt = torch.optim.Adam(save_m.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    best_state, best_vf1, pat = None, -1.0, CFG.patience
    for ep in range(1, CFG.epochs + 1):
        save_m.train(); save_opt.zero_grad()
        F.cross_entropy(save_m(A, X_tr), y_tr, weight=cw,
                        label_smoothing=CFG.label_smoothing).backward()
        torch.nn.utils.clip_grad_norm_(save_m.parameters(), 5.0)
        save_opt.step()
        vf1, _, _ = eval_at(get_probs(save_m, A, X_va), y_va_np)
        if vf1 > best_vf1 + 1e-4:
            best_vf1 = vf1
            best_state = {k: v.detach().cpu().clone() for k, v in save_m.state_dict().items()}
            pat = CFG.patience
        else:
            pat -= 1
            if pat <= 0:
                break
    if best_state:
        save_m.load_state_dict(best_state)
    torch.save({
        "state_dict":    save_m.state_dict(),
        "num_words":     num_words,
        "hidden_dim":    CFG.hidden_dim,
        "dropout":       CFG.dropout,
        "residual_alpha": CFG.residual_alpha,
        "model_type":    "base",
        "trial":         CFG.name,
        "threshold":     thr["t"],
    }, TUNED_MODEL_DIR / "textgcn_tuned.pt")

    print(f"\n✅ Artifacts → {TUNED_MODEL_DIR}")
    print(f"✅ Metrics   → {REPORTS_DIR / 'metrics_textgcn_tuned.csv'}")


if __name__ == "__main__":
    main()
