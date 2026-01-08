# src/evaluate.py
import argparse
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

from src.config import PATHS

def load_splits():
    with open(PATHS.data_processed / "splits.json", "r", encoding="utf-8") as f:
        return json.load(f)

def subset_by_ids(df: pd.DataFrame, ids):
    ids = set(map(str, ids))
    return df[df["id"].astype(str).isin(ids)].copy()

def score_to_pred(score, model_name):
    # LinearSVC uses decision_function: threshold at 0.0
    if "svm" in model_name:
        return (score >= 0.0).astype(int)
    return (score >= 0.5).astype(int)

def eval_emscad(y_true, score, pred):
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, score),
        "pr_auc": average_precision_score(y_true, score),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baselines"], default="baselines")
    args = ap.parse_args()

    PATHS.reports.mkdir(parents=True, exist_ok=True)

    em = pd.read_csv(PATHS.data_processed / "emscad.csv")
    ob = pd.read_csv(PATHS.data_processed / "openbay.csv")
    splits = load_splits()

    train = subset_by_ids(em, splits["train_ids"])
    val   = subset_by_ids(em, splits["val_ids"])
    test  = subset_by_ids(em, splits["test_ids"])

    vec = joblib.load(PATHS.models_baselines / "vectorizer.joblib")
    X_test = vec.transform(test["text"])
    y_test = test["fraudulent"].astype(int).values

    # OpenDataBay robustness set
    X_ob = vec.transform(ob["text"])

    rows = []
    for model_path in sorted(PATHS.models_baselines.glob("tfidf_*.joblib")):
        name = model_path.stem
        model = joblib.load(model_path)

        if hasattr(model, "predict_proba"):
            score = model.predict_proba(X_test)[:, 1]
            ob_score = model.predict_proba(X_ob)[:, 1]
        else:
            score = model.decision_function(X_test)
            ob_score = model.decision_function(X_ob)

        pred = score_to_pred(score, name)
        metrics = eval_emscad(y_test, score, pred)

        # OpenDataBay: fraud-only in your file (all 1s)
        # So "recall" = fraction predicted as fraud at chosen threshold.
        ob_pred = score_to_pred(ob_score, name)
        openbay_recall = float(np.mean(ob_pred))

        rows.append({
            "model": name,
            **metrics,
            "openbay_recall": openbay_recall,
            "openbay_mean_score": float(np.mean(ob_score)),
            "openbay_median_score": float(np.median(ob_score)),
        })

    out = pd.DataFrame(rows).sort_values("f1", ascending=False)
    out_path = PATHS.reports / "metrics_baselines.csv"
    out.to_csv(out_path, index=False)

    print("✅ Saved:", out_path)
    print(out)

if __name__ == "__main__":
    main()
