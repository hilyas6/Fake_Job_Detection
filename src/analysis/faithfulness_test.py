"""
SHAP / Occlusion Faithfulness Test
====================================
Tests whether the model's explanations are faithful to its decisions.

Method (occlusion-based faithfulness):
  For each sampled test document:
    1. Record original fake probability  p0
    2. Identify top-K features by TF-IDF weight (proxy for 'important features')
    3. Remove top-K features one-by-one and re-score → measure cumulative drop
    4. Repeat with K random features as a control
  A faithful model should show a much larger drop when *top* features are removed
  than when *random* features are removed.

Outputs
-------
  reports/faithfulness_results.csv       – per-sample results
  reports/faithfulness_summary.txt       – plain-text dissertation table
"""
import json
import random
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

from tuned_models.tune_textgcn import ImprovedWordGCN, to_sparse, tokenize


def load_model_and_vec():
    device = torch.device("cpu")
    vec    = joblib.load(TUNED_MODEL_DIR / "vectorizer_tuned.joblib")
    graph  = torch.load(TUNED_MODEL_DIR / "graph_cache_tuned.pt",
                        map_location="cpu", weights_only=False)
    ckpt   = torch.load(TUNED_MODEL_DIR / "textgcn_tuned.pt",
                        map_location="cpu", weights_only=False)

    A = torch.sparse_coo_tensor(
        graph["A_norm_indices"],
        graph["A_norm_values"],
        tuple(graph["A_norm_size"]),
    ).coalesce().to(device)

    model = ImprovedWordGCN(
        num_words      = int(ckpt["num_words"]),
        hidden_dim     = int(ckpt["hidden_dim"]),
        dropout        = float(ckpt["dropout"]),
        residual_alpha = float(ckpt.get("residual_alpha", 0.7)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    threshold = float(ckpt.get("threshold", 0.48))
    return model, vec, A, device, threshold


@torch.no_grad()
def predict_proba(model, vec, A, device, texts: list) -> np.ndarray:
    X = to_sparse(vec.transform(texts)).to(device)
    logits = model(A, X)
    return F.softmax(logits, dim=1)[:, 1].cpu().numpy()


def get_top_features(vec, text: str, k: int) -> list[str]:
    """Return the top-k vocabulary features by TF-IDF weight for this text."""
    x   = vec.transform([text]).tocoo()
    fnames = vec.get_feature_names_out()
    pairs  = sorted(zip(x.col, x.data), key=lambda p: p[1], reverse=True)
    return [str(fnames[col]) for col, _ in pairs[:k]]


def mask_features(text: str, features: list[str]) -> str:
    """Replace all occurrences of each feature token/phrase with a space."""
    import re
    result = text
    for feat in features:
        result = re.sub(rf"\b{re.escape(feat)}\b", " ", result, flags=re.IGNORECASE)
    return result


def faithfulness_drop(
    model, vec, A, device, text: str, p0: float,
    k_values: list[int] = [1, 3, 5, 10],
) -> dict:
    """
    Measure fake-probability drop when removing top-K vs random-K features.
    Returns dict with keys top_{k} and rand_{k} for each k in k_values.
    """
    x = vec.transform([text]).tocoo()
    fnames = vec.get_feature_names_out()
    all_features = [str(fnames[col]) for col in x.col]

    if not all_features:
        return {}

    top_features  = get_top_features(vec, text, max(k_values))
    rand_features = random.sample(all_features, min(max(k_values), len(all_features)))

    result = {"p0": p0}
    for k in k_values:
        if k > len(all_features):
            continue

        # Top-k removal
        masked_top  = mask_features(text, top_features[:k])
        p_top       = predict_proba(model, vec, A, device, [masked_top])[0]
        result[f"top_{k}_prob"]  = float(p_top)
        result[f"top_{k}_drop"]  = float(p0 - p_top)

        # Random-k removal (control)
        masked_rand = mask_features(text, rand_features[:k])
        p_rand      = predict_proba(model, vec, A, device, [masked_rand])[0]
        result[f"rand_{k}_prob"] = float(p_rand)
        result[f"rand_{k}_drop"] = float(p0 - p_rand)

    return result


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(42)

    print("Loading model...")
    model, vec, A, device, threshold = load_model_and_vec()

    em = pd.read_csv(DATA_PROCESSED / "emscad.csv")
    with open(DATA_PROCESSED / "splits.json", encoding="utf-8") as f:
        splits = json.load(f)
    test_df = em[em["id"].astype(str).isin(set(map(str, splits["test_ids"])))].copy()

    # Sample: 100 fake + 100 real (balanced) from test set
    fake_df = test_df[test_df["fraudulent"] == 1].sample(
        min(100, (test_df["fraudulent"] == 1).sum()), random_state=42)
    real_df = test_df[test_df["fraudulent"] == 0].sample(100, random_state=42)
    sample_df = pd.concat([fake_df, real_df]).reset_index(drop=True)
    print(f"Evaluating faithfulness on {len(sample_df)} samples "
          f"({len(fake_df)} fake + {len(real_df)} real)...")

    # Get baseline probabilities
    p0_all = predict_proba(model, vec, A, device, sample_df["text"].tolist())
    sample_df["p0"] = p0_all

    k_values = [1, 3, 5, 10]
    rows = []
    for i, row in sample_df.iterrows():
        if i % 20 == 0:
            print(f"  {i}/{len(sample_df)}...")
        res = faithfulness_drop(model, vec, A, device, row["text"],
                                float(row["p0"]), k_values=k_values)
        if res:
            res["fraudulent"] = int(row["fraudulent"])
            rows.append(res)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(REPORTS_DIR / "faithfulness_results.csv", index=False)

    # Summary
    lines = [
        "FAITHFULNESS TEST – TextGCN Tuned",
        "=" * 55,
        "Method: Remove top-K TF-IDF features vs random-K features",
        "Metric: drop in fake probability (higher = more faithful)",
        "",
        f"{'K':>4} | {'Top-K drop':>12} | {'Random-K drop':>13} | {'Ratio':>7}",
        "-" * 55,
    ]
    for k in k_values:
        col_top  = f"top_{k}_drop"
        col_rand = f"rand_{k}_drop"
        if col_top not in results_df.columns:
            continue
        avg_top  = results_df[col_top].mean()
        avg_rand = results_df[col_rand].mean()
        ratio    = avg_top / (avg_rand + 1e-9)
        lines.append(f"{k:>4} | {avg_top:>12.4f} | {avg_rand:>13.4f} | {ratio:>7.2f}x")
    lines.append("-" * 55)

    # Breakdown by fake vs real
    lines.append("\nFake samples only:")
    fake_res = results_df[results_df["fraudulent"] == 1]
    for k in k_values:
        col_top  = f"top_{k}_drop"
        col_rand = f"rand_{k}_drop"
        if col_top not in results_df.columns or fake_res.empty:
            continue
        avg_top  = fake_res[col_top].mean()
        avg_rand = fake_res[col_rand].mean()
        lines.append(f"  K={k}: top-K drop={avg_top:.4f}  rand-K drop={avg_rand:.4f}")

    lines.append("\nReal samples only:")
    real_res = results_df[results_df["fraudulent"] == 0]
    for k in k_values:
        col_top  = f"top_{k}_drop"
        col_rand = f"rand_{k}_drop"
        if col_top not in results_df.columns or real_res.empty:
            continue
        avg_top  = real_res[col_top].mean()
        avg_rand = real_res[col_rand].mean()
        lines.append(f"  K={k}: top-K drop={avg_top:.4f}  rand-K drop={avg_rand:.4f}")

    lines.append(
        "\nInterpretation: A ratio > 1.0 means removing the model's most "
        "influential features causes a larger prediction change than removing "
        "random features, confirming that explanations reflect real model decisions."
    )

    summary = "\n".join(lines)
    print("\n" + summary)
    summary_path = REPORTS_DIR / "faithfulness_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"\nSaved: {REPORTS_DIR / 'faithfulness_results.csv'}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
