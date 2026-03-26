"""
Error Analysis – Tuned TextGCN
================================
Loads the tuned TextGCN model, runs inference on the EMSCAD test set, and
produces a detailed breakdown of:
  - False Positives  (real jobs predicted as fake)
  - False Negatives  (fake jobs predicted as real)

Characterises each error class by text length, word count, presence of key
fields, and (if available) industry/function labels.

Outputs
-------
  reports/error_analysis_fp.csv          – false positive records + metadata
  reports/error_analysis_fn.csv          – false negative records + metadata
  reports/error_analysis_summary.txt     – plain-text summary for dissertation
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA_PROCESSED  = ROOT / "data" / "processed"
TUNED_MODEL_DIR = ROOT / "models" / "tuned" / "textgcn_tuned"
REPORTS_DIR     = ROOT / "reports"

# ── local imports ────────────────────────────────────────────────────────────
from tuned_models.tune_textgcn import (
    ImprovedWordGCN,
    to_sparse,
    tokenize,
)


def load_model_and_data():
    device = torch.device("cpu")

    # ── vectorizer + graph ───────────────────────────────────────────────────
    vec = joblib.load(TUNED_MODEL_DIR / "vectorizer_tuned.joblib")
    graph = torch.load(TUNED_MODEL_DIR / "graph_cache_tuned.pt",
                       map_location="cpu", weights_only=False)
    ckpt  = torch.load(TUNED_MODEL_DIR / "textgcn_tuned.pt",
                       map_location="cpu", weights_only=False)

    A = torch.sparse_coo_tensor(
        graph["A_norm_indices"],
        graph["A_norm_values"],
        tuple(graph["A_norm_size"]),
    ).coalesce().to(device)

    model = ImprovedWordGCN(
        num_words     = int(ckpt["num_words"]),
        hidden_dim    = int(ckpt["hidden_dim"]),
        dropout       = float(ckpt["dropout"]),
        residual_alpha= float(ckpt.get("residual_alpha", 0.7)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    threshold = float(ckpt.get("threshold", 0.48))

    # ── data ─────────────────────────────────────────────────────────────────
    em = pd.read_csv(DATA_PROCESSED / "emscad.csv")
    with open(DATA_PROCESSED / "splits.json", encoding="utf-8") as f:
        splits = json.load(f)
    test_df = em[em["id"].astype(str).isin(set(map(str, splits["test_ids"])))].copy()

    X_te_s = vec.transform(test_df["text"])
    X_te   = to_sparse(X_te_s).to(device)
    y_te   = test_df["fraudulent"].astype(int).values

    # ── inference ────────────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(A, X_te)
        probs  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()

    preds = (probs >= threshold).astype(int)
    return test_df, y_te, preds, probs, threshold


def characterise(df: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    """Add diagnostic columns to a sub-DataFrame."""
    df = df.copy()
    df["fake_prob"]   = probs
    df["text_len"]    = df["text"].str.len()
    df["word_count"]  = df["text"].str.split().str.len()
    df["has_sep"]     = df["text"].str.contains(r"\[SEP\]", regex=True)
    df["has_url"]     = df["text"].str.contains(r"__URL__", regex=False)
    df["has_email"]   = df["text"].str.contains(r"__EMAIL__", regex=False)
    df["has_phone"]   = df["text"].str.contains(r"__PHONE__", regex=False)
    df["has_money"]   = df["text"].str.contains(r"__MONEY__", regex=False)
    df["n_sep_tokens"] = df["text"].str.count(r"\[SEP\]")
    return df


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model and data...")
    test_df, y_te, preds, probs, threshold = load_model_and_data()
    print(f"Test set: {len(test_df)} samples | threshold={threshold:.2f}")

    from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
    f1   = f1_score(y_te, preds, zero_division=0)
    prec = precision_score(y_te, preds, zero_division=0)
    rec  = recall_score(y_te, preds, zero_division=0)
    cm   = confusion_matrix(y_te, preds)

    print(f"F1={f1:.4f}  P={prec:.4f}  R={rec:.4f}")
    print(f"Confusion matrix:\n  TN={cm[0,0]}  FP={cm[0,1]}\n  FN={cm[1,0]}  TP={cm[1,1]}")

    # ── split error types ────────────────────────────────────────────────────
    fp_mask = (preds == 1) & (y_te == 0)   # real predicted as fake
    fn_mask = (preds == 0) & (y_te == 1)   # fake predicted as real
    tp_mask = (preds == 1) & (y_te == 1)
    tn_mask = (preds == 0) & (y_te == 0)

    fp_df = characterise(test_df[fp_mask].reset_index(drop=True), probs[fp_mask])
    fn_df = characterise(test_df[fn_mask].reset_index(drop=True), probs[fn_mask])
    tp_df = characterise(test_df[tp_mask].reset_index(drop=True), probs[tp_mask])
    tn_df = characterise(test_df[tn_mask].reset_index(drop=True), probs[tn_mask])

    fp_df.to_csv(REPORTS_DIR / "error_analysis_fp.csv", index=False)
    fn_df.to_csv(REPORTS_DIR / "error_analysis_fn.csv", index=False)

    # ── summary stats ────────────────────────────────────────────────────────
    def stats(df, name):
        lines = [f"\n{'='*50}", f"  {name}  (n={len(df)})", "="*50]
        if df.empty:
            lines.append("  (none)")
            return "\n".join(lines)
        lines.append(f"  Fake probability : mean={df['fake_prob'].mean():.3f}  "
                     f"min={df['fake_prob'].min():.3f}  max={df['fake_prob'].max():.3f}")
        lines.append(f"  Text length (chars): mean={df['text_len'].mean():.0f}  "
                     f"median={df['text_len'].median():.0f}")
        lines.append(f"  Word count         : mean={df['word_count'].mean():.0f}  "
                     f"median={df['word_count'].median():.0f}")
        lines.append(f"  Has URL token      : {df['has_url'].mean()*100:.1f}%")
        lines.append(f"  Has EMAIL token    : {df['has_email'].mean()*100:.1f}%")
        lines.append(f"  Has MONEY token    : {df['has_money'].mean()*100:.1f}%")
        lines.append(f"  Avg [SEP] tokens   : {df['n_sep_tokens'].mean():.1f}")

        # Short vs long text breakdown
        short = df[df["text_len"] < 500]
        long_  = df[df["text_len"] >= 1500]
        lines.append(f"  Short (<500 chars) : {len(short)} ({len(short)/len(df)*100:.1f}%)")
        lines.append(f"  Long  (>=1500 chars): {len(long_)} ({len(long_)/len(df)*100:.1f}%)")

        # Show top-5 examples
        lines.append("\n  Top-5 examples (sorted by fake_prob desc):")
        top5 = df.nlargest(5, "fake_prob")[["fake_prob", "text_len", "text"]]
        for i, row in top5.iterrows():
            snippet = str(row["text"])[:120].replace("\n", " ")
            lines.append(f"    [{i}] prob={row['fake_prob']:.3f} len={row['text_len']}  \"{snippet}...\"")
        return "\n".join(lines)

    # Compare FP/FN to TP/TN to understand what makes them hard
    summary_lines = [
        "ERROR ANALYSIS – TextGCN Tuned (3-seed ensemble)",
        f"Test set size : {len(test_df)}",
        f"Threshold     : {threshold}",
        f"F1={f1:.4f}  Precision={prec:.4f}  Recall={rec:.4f}",
        f"\nConfusion matrix:",
        f"  True Negative  (correct real) : {cm[0,0]}",
        f"  False Positive (real as fake) : {cm[0,1]}",
        f"  False Negative (fake as real) : {cm[1,0]}",
        f"  True Positive  (correct fake) : {cm[1,1]}",
    ]

    for name, df in [
        ("FALSE POSITIVES – Real jobs misclassified as FAKE", fp_df),
        ("FALSE NEGATIVES – Fake jobs misclassified as REAL", fn_df),
        ("TRUE POSITIVES  – Correctly detected FAKE jobs",    tp_df),
        ("TRUE NEGATIVES  – Correctly classified REAL jobs",  tn_df),
    ]:
        summary_lines.append(stats(df, name))

    # FP vs TN comparison  (what makes FPs look fake?)
    if not fp_df.empty and not tn_df.empty:
        summary_lines.append("\n" + "="*50)
        summary_lines.append("FP vs TN – Why were real jobs flagged as fake?")
        summary_lines.append("="*50)
        for col in ["text_len", "word_count", "has_url", "has_email", "has_money", "n_sep_tokens"]:
            fp_v = fp_df[col].mean()
            tn_v = tn_df[col].mean()
            summary_lines.append(f"  {col:<20}: FP={fp_v:.2f}  TN={tn_v:.2f}")

    # FN vs TP comparison  (what makes FNs look real?)
    if not fn_df.empty and not tp_df.empty:
        summary_lines.append("\n" + "="*50)
        summary_lines.append("FN vs TP – Why were fake jobs missed?")
        summary_lines.append("="*50)
        for col in ["text_len", "word_count", "has_url", "has_email", "has_money", "n_sep_tokens"]:
            fn_v = fn_df[col].mean()
            tp_v = tp_df[col].mean()
            summary_lines.append(f"  {col:<20}: FN={fn_v:.2f}  TP={tp_v:.2f}")

    summary = "\n".join(summary_lines)
    print(summary)

    summary_path = REPORTS_DIR / "error_analysis_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"\nSummary saved to: {summary_path}")
    print(f"FP records : {REPORTS_DIR / 'error_analysis_fp.csv'}")
    print(f"FN records : {REPORTS_DIR / 'error_analysis_fn.csv'}")


if __name__ == "__main__":
    main()
