"""
evaluate_only.py — Quick display of all model metrics without retraining.

Reads saved CSV results from reports/ and reports/tuned/ and prints a
ranked comparison table. Run this any time to check the current standings.

Usage:
    python evaluate_only.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPORTS = Path(__file__).resolve().parent / "reports"

# (display_name, baseline_csv, tuned_csv)  —  paths relative to REPORTS
MODELS = [
    ("TextGCN (Tuned Ensemble)",  None,                               "tuned/metrics_textgcn_tuned.csv"),
    ("TextGCN (Improved)",        "metrics_textgcn_improved.csv",     None),
    ("LightGBM",                  "metrics_lightgbm.csv",             "tuned/metrics_lightgbm.csv"),
    ("XGBoost",                   "metrics_xgboost.csv",              "tuned/metrics_xgboost.csv"),
    ("Logistic Regression",       "metrics_logistic_regression.csv",  "tuned/metrics_logistic_regression.csv"),
    ("Random Forest",             "metrics_random_forest.csv",        "tuned/metrics_random_forest.csv"),
    ("DistilBERT",                "metrics_distilbert.csv",           "tuned/metrics_distilbert.csv"),
    ("BiLSTM",                    "metrics_bilstm.csv",               "tuned/metrics_bilstm.csv"),
    ("Naive Bayes",               "metrics_naive_bayes.csv",          "tuned/metrics_naive_bayes.csv"),
]

KEY_COLS = ["emscad_test_f1", "emscad_test_precision", "emscad_test_recall",
            "openbay_recall", "threshold"]
RENAME   = {
    "emscad_test_f1":        "F1",
    "emscad_test_precision": "Precision",
    "emscad_test_recall":    "Recall",
    "openbay_recall":        "OOD Recall",
    "threshold":             "Threshold",
}
SEP = "-" * 100


def _load(rel: str) -> dict | None:
    path = REPORTS / rel
    try:
        df = pd.read_csv(path)
        return df.iloc[0].to_dict() if not df.empty else None
    except FileNotFoundError:
        return None


def _fmt(v) -> str:
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v) if v is not None else "—"


def main() -> None:
    print(SEP)
    print("  FAKE JOB DETECTION — MODEL EVALUATION SUMMARY")
    print(SEP)

    rows = []
    for name, base_file, tuned_file in MODELS:
        for label, file in [("Baseline", base_file), ("Tuned", tuned_file)]:
            if file is None:
                continue
            data = _load(file)
            if data is None:
                continue
            rows.append({"Model": name, "Version": label, **{k: data.get(k) for k in KEY_COLS}})

    if not rows:
        print("No metrics found. Train the models first (see README.md).")
        return

    df = pd.DataFrame(rows).sort_values("emscad_test_f1", ascending=False).reset_index(drop=True)

    # Header
    col_w = {"Model": 30, "Version": 9, "F1": 8, "Precision": 9, "Recall": 8, "OOD Recall": 10, "Threshold": 10}
    df = df.rename(columns=RENAME)
    header = "  " + "".join(c.ljust(col_w.get(c, 10)) for c in col_w)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for rank, (_, row) in enumerate(df.iterrows(), 1):
        line = f"{rank:>2}  "
        line += str(row["Model"]).ljust(col_w["Model"])
        line += str(row["Version"]).ljust(col_w["Version"])
        for col in ["F1", "Precision", "Recall", "OOD Recall", "Threshold"]:
            line += _fmt(row.get(col)).ljust(col_w[col])
        # Mark production model
        if "Tuned Ensemble" in str(row["Model"]) and row["Version"] == "Tuned":
            line += " ← production"
        print(line)

    print(SEP)

    # Quick summary stats
    best = df.iloc[0]
    print(f"\n  Best F1        : {best['Model']} ({best['Version']})  →  {_fmt(best['F1'])}")
    tuned_only = df[df["Version"] == "Tuned"]
    if not tuned_only.empty:
        best_tuned = tuned_only.iloc[0]
        print(f"  Best Tuned F1  : {best_tuned['Model']}  →  {_fmt(best_tuned['F1'])}")
    ood = df[df["OOD Recall"].apply(lambda x: x is not None)].copy()
    ood["OOD Recall"] = pd.to_numeric(ood["OOD Recall"], errors="coerce")
    if not ood.empty:
        best_ood = ood.sort_values("OOD Recall", ascending=False).iloc[0]
        print(f"  Best OOD Recall: {best_ood['Model']} ({best_ood['Version']})  →  {_fmt(best_ood['OOD Recall'])}")
    print()


if __name__ == "__main__":
    main()
