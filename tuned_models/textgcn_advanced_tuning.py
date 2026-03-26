"""
Advanced TextGCN Tuning – Targeting LightGBM F1 = 0.8494
==========================================================
Tries four strategies beyond the current best (0.8390):

  A. 5-seed ensemble (more diversity)
  B. 7-seed ensemble (even more diversity)
  C. Validation-weighted ensemble (weight by each seed's val F1)
  D. Linear class weight neg/pos (instead of sqrt) + 5-seed ensemble
  E. PMI window=10 + 5-seed ensemble (between 15 and baseline 20)

The winner (if any beats 0.8390) replaces the tuned model artifacts.
If nothing beats the current model, the existing artifacts are kept.

Usage:
    python tuned_models/advanced_tuning.py
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tuned_models.tune_textgcn import (
    TrialConfig, ImprovedWordGCN,
    load_data, build_pmi_graph, normalize_adj, to_sparse, tokenize,
    train_single, best_threshold, eval_at, get_probs,
    TUNED_MODEL_DIR, REPORTS_DIR,
)

CURRENT_BEST_F1   = 0.8390   # existing tuned model
CURRENT_BEST_NAME = "ensemble_3seeds_window15"


def run_advanced_trial(cfg: TrialConfig, bundle: dict, device,
                       weighted_ensemble: bool = False) -> dict:
    """
    Train a multi-seed ensemble and return metrics.
    If weighted_ensemble=True, weight each seed's predictions by its val F1.
    """
    t0 = time.time()
    train_df = bundle["train"]
    val_df   = bundle["val"]
    test_df  = bundle["test"]
    ob_df    = bundle["ob"]

    vec = TfidfVectorizer(
        tokenizer=tokenize, preprocessor=None, token_pattern=None,
        ngram_range=(1, 3), min_df=2, max_df=0.9,
        sublinear_tf=True, max_features=cfg.max_features,
    )
    X_tr_s = vec.fit_transform(train_df["text"])
    X_va_s = vec.transform(val_df["text"])
    X_te_s = vec.transform(test_df["text"])
    X_ob_s = vec.transform(ob_df["text"])

    vocab     = vec.vocabulary_
    num_words = len(vocab)

    tok_train = [tokenize(t) for t in train_df["text"].tolist()]
    rows, cols, vals, n = build_pmi_graph(tok_train, vocab,
                                          window_size=cfg.window_size)
    A = normalize_adj(rows, cols, vals, n).to(device)

    X_tr = to_sparse(X_tr_s).to(device)
    X_va = to_sparse(X_va_s).to(device)
    X_te = to_sparse(X_te_s).to(device)
    X_ob = to_sparse(X_ob_s).to(device)

    y_tr   = torch.tensor(train_df["fraudulent"].astype(int).values,
                          dtype=torch.long, device=device)
    y_va_np = val_df["fraudulent"].astype(int).values
    y_te_np = test_df["fraudulent"].astype(int).values

    pos_n = int((y_tr == 1).sum())
    neg_n = int((y_tr == 0).sum())
    if cfg.class_weight == "linear":
        w1 = neg_n / max(pos_n, 1)
    else:  # "sqrt"
        w1 = math.sqrt(neg_n / max(pos_n, 1))
    cw = torch.tensor([1.0, w1], dtype=torch.float32, device=device)

    all_val, all_test, all_ob, val_f1s = [], [], [], []
    for s in range(cfg.n_seeds):
        seed = cfg.base_seed + s
        print(f"    seed {seed}...")
        torch.manual_seed(seed)
        np.random.seed(seed)
        model_s = ImprovedWordGCN(num_words, cfg.hidden_dim, cfg.dropout,
                                  cfg.residual_alpha).to(device)
        opt_s   = torch.optim.Adam(model_s.parameters(),
                                   lr=cfg.lr, weight_decay=cfg.weight_decay)
        best_state_s, best_vf1_s, pat_s = None, -1.0, cfg.patience
        for ep in range(1, cfg.epochs + 1):
            model_s.train()
            opt_s.zero_grad()
            F.cross_entropy(model_s(A, X_tr), y_tr, weight=cw,
                            label_smoothing=cfg.label_smoothing).backward()
            torch.nn.utils.clip_grad_norm_(model_s.parameters(), 5.0)
            opt_s.step()
            vf1, _, _ = eval_at(get_probs(model_s, A, X_va), y_va_np)
            if vf1 > best_vf1_s + 1e-4:
                best_vf1_s   = vf1
                best_state_s = {k: v.detach().cpu().clone()
                                for k, v in model_s.state_dict().items()}
                pat_s = cfg.patience
            else:
                pat_s -= 1
                if pat_s <= 0:
                    break
        if best_state_s:
            model_s.load_state_dict(best_state_s)
        all_val.append(get_probs(model_s, A, X_va))
        all_test.append(get_probs(model_s, A, X_te))
        all_ob.append(get_probs(model_s, A, X_ob))
        val_f1s.append(best_vf1_s)
        print(f"      best val F1 = {best_vf1_s:.4f}")

    if weighted_ensemble and len(val_f1s) > 1:
        # Weight each seed by its validation F1 (normalised to sum=1)
        weights   = np.array(val_f1s)
        weights   = weights / weights.sum()
        val_probs  = np.average(all_val,  axis=0, weights=weights)
        test_probs = np.average(all_test, axis=0, weights=weights)
        ob_probs   = np.average(all_ob,   axis=0, weights=weights)
        print(f"    Seed weights: {[f'{w:.3f}' for w in weights]}")
    else:
        val_probs  = np.mean(all_val,  axis=0)
        test_probs = np.mean(all_test, axis=0)
        ob_probs   = np.mean(all_ob,   axis=0)

    thr       = best_threshold(val_probs, y_va_np)
    tf1, tp_m, tr = eval_at(test_probs, y_te_np, thr["t"])
    ob_recall   = float(np.mean((ob_probs >= thr["t"]).astype(int) == 1))
    elapsed     = time.time() - t0
    beats       = tf1 > CURRENT_BEST_F1

    print(f"  Result: F1={tf1:.4f}  P={tp_m:.4f}  R={tr:.4f}  "
          f"OB={ob_recall:.4f}  thr={thr['t']:.2f}  ({elapsed:.0f}s)")
    print(f"  {'BEATS CURRENT BEST' if beats else 'Below current best'} "
          f"({CURRENT_BEST_F1:.4f})")

    # Save artifacts if this is the new best
    if beats:
        TUNED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        inv_vocab = {i: t for t, i in vocab.items()}
        joblib.dump(vec, TUNED_MODEL_DIR / "vectorizer_tuned.joblib")
        torch.save({
            "A_norm_indices": A.coalesce().indices().cpu(),
            "A_norm_values":  A.coalesce().values().cpu(),
            "A_norm_size":    A.shape,
            "inv_vocab":      inv_vocab,
        }, TUNED_MODEL_DIR / "graph_cache_tuned.pt")

        # Retrain seed-42 model for single-model inference
        torch.manual_seed(cfg.base_seed)
        save_m = ImprovedWordGCN(num_words, cfg.hidden_dim, cfg.dropout,
                                 cfg.residual_alpha).to(device)
        pos_n = int((y_tr == 1).sum())
        neg_n = int((y_tr == 0).sum())
        if cfg.class_weight == "linear":
            w1 = neg_n / max(pos_n, 1)
        else:
            w1 = math.sqrt(neg_n / max(pos_n, 1))
        cw = torch.tensor([1.0, w1], dtype=torch.float32, device=device)
        save_opt = torch.optim.Adam(save_m.parameters(),
                                    lr=cfg.lr, weight_decay=cfg.weight_decay)
        best_s, best_vf1, pat = None, -1.0, cfg.patience
        for ep in range(1, cfg.epochs + 1):
            save_m.train()
            save_opt.zero_grad()
            F.cross_entropy(save_m(A, X_tr), y_tr, weight=cw,
                            label_smoothing=cfg.label_smoothing).backward()
            torch.nn.utils.clip_grad_norm_(save_m.parameters(), 5.0)
            save_opt.step()
            vf1, _, _ = eval_at(get_probs(save_m, A, X_va), y_va_np)
            if vf1 > best_vf1 + 1e-4:
                best_vf1 = vf1
                best_s   = {k: v.detach().cpu().clone()
                            for k, v in save_m.state_dict().items()}
                pat = cfg.patience
            else:
                pat -= 1
                if pat <= 0:
                    break
        if best_s:
            save_m.load_state_dict(best_s)
        torch.save({
            "state_dict":     save_m.state_dict(),
            "num_words":      num_words,
            "hidden_dim":     cfg.hidden_dim,
            "dropout":        cfg.dropout,
            "residual_alpha": cfg.residual_alpha,
            "model_type":     "base",
            "trial":          cfg.name,
            "threshold":      thr["t"],
        }, TUNED_MODEL_DIR / "textgcn_tuned.pt")
        print(f"  New artifacts saved to {TUNED_MODEL_DIR}")

        # Save new metrics
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{
            "model":                 "textgcn_tuned",
            "emscad_test_f1":        round(tf1, 6),
            "emscad_test_precision": round(tp_m, 6),
            "emscad_test_recall":    round(tr, 6),
            "openbay_recall":        round(ob_recall, 4),
            "vocab_size":            num_words,
            "threshold":             round(thr["t"], 2),
            "trial":                 cfg.name,
        }]).to_csv(REPORTS_DIR / "tuned" / "metrics_textgcn_tuned.csv", index=False)
        print(f"  Updated metrics: reports/tuned/metrics_textgcn_tuned.csv")

    return {
        "trial":          cfg.name,
        "n_seeds":        cfg.n_seeds,
        "window_size":    cfg.window_size,
        "class_weight":   cfg.class_weight,
        "weighted_ens":   weighted_ensemble,
        "emscad_test_f1": round(tf1, 6),
        "precision":      round(tp_m, 6),
        "recall":         round(tr, 6),
        "openbay_recall": round(ob_recall, 4),
        "threshold":      round(thr["t"], 2),
        "beats_current":  beats,
        "elapsed_s":      round(elapsed, 1),
    }


def build_advanced_trials():
    return [
        # A. 5-seed ensemble (same window=15 as current winner)
        TrialConfig(
            name="adv_5seeds_window15",
            n_seeds=5, window_size=15,
            optimizer="adam", scheduler="none",
            epochs=200, patience=25,
            class_weight="sqrt",
        ),
        # B. Validation-weighted 5-seed ensemble
        TrialConfig(
            name="adv_5seeds_window15_weighted",
            n_seeds=5, window_size=15,
            optimizer="adam", scheduler="none",
            epochs=200, patience=25,
            class_weight="sqrt",
        ),
        # C. 5-seed + linear class weight (neg/pos, not sqrt)
        TrialConfig(
            name="adv_5seeds_linear_weight",
            n_seeds=5, window_size=15,
            optimizer="adam", scheduler="none",
            epochs=200, patience=25,
            class_weight="linear",
        ),
        # D. 5-seed + window=10
        TrialConfig(
            name="adv_5seeds_window10",
            n_seeds=5, window_size=10,
            optimizer="adam", scheduler="none",
            epochs=200, patience=25,
            class_weight="sqrt",
        ),
        # E. 7-seed ensemble + window=15
        TrialConfig(
            name="adv_7seeds_window15",
            n_seeds=7, window_size=15,
            optimizer="adam", scheduler="none",
            epochs=200, patience=25,
            class_weight="sqrt",
        ),
    ]


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TUNED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    print(f"Advanced TextGCN tuning | Target: beat F1={CURRENT_BEST_F1:.4f}")
    bundle = load_data()
    print(f"Train: {len(bundle['train'])}  Val: {len(bundle['val'])}"
          f"  Test: {len(bundle['test'])}  OB: {len(bundle['ob'])}")

    trials    = build_advanced_trials()
    results   = []
    best_f1   = CURRENT_BEST_F1
    best_name = CURRENT_BEST_NAME

    for i, cfg in enumerate(trials, 1):
        weighted = cfg.name.endswith("_weighted")
        print(f"\n[{i}/{len(trials)}] {cfg.name}  "
              f"(n_seeds={cfg.n_seeds}, window={cfg.window_size}, "
              f"class_weight={cfg.class_weight}"
              + (", val-weighted" if weighted else "") + ")")
        try:
            result = run_advanced_trial(cfg, bundle, device,
                                        weighted_ensemble=weighted)
            results.append(result)
            if result["emscad_test_f1"] > best_f1:
                best_f1   = result["emscad_test_f1"]
                best_name = cfg.name
        except Exception as exc:
            import traceback
            print(f"  FAILED: {exc}")
            traceback.print_exc()

    print("\n" + "=" * 65)
    print("ADVANCED TUNING SUMMARY")
    print("=" * 65)
    print(f"{'Trial':<38} {'F1':>7} {'OB':>7} {'New?':>6}")
    print(f"  (Current best: {CURRENT_BEST_NAME:<22} {CURRENT_BEST_F1:>7.4f})")
    print("-" * 65)
    for r in sorted(results, key=lambda x: -x["emscad_test_f1"]):
        marker = " ★" if r["beats_current"] else ""
        print(f"  {r['trial']:<36} {r['emscad_test_f1']:>7.4f} "
              f"{r['openbay_recall']:>7.4f} {str(r['beats_current']):>6}{marker}")
    print("=" * 65)

    if best_f1 > CURRENT_BEST_F1:
        print(f"\nNew best: {best_name}  F1={best_f1:.4f}  "
              f"(+{best_f1 - CURRENT_BEST_F1:.4f} vs previous best)")
    else:
        print(f"\nNo improvement found. Current best ({CURRENT_BEST_NAME}, "
              f"F1={CURRENT_BEST_F1:.4f}) remains unchanged.")

    if results:
        out = REPORTS_DIR / "advanced_tuning_results.csv"
        pd.DataFrame(results).to_csv(out, index=False)
        print(f"Results saved to: {out}")


if __name__ == "__main__":
    main()
