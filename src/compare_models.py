"""Aggregate baseline and tuned metrics into a single comparison CSV."""
from __future__ import annotations

import pandas as pd
from src.config import PATHS

_DROP = {"vocab_size", "pmi_window_size", "pmi_threshold", "residual_alpha", "trial"}

# (display_name, baseline_csv_relative_to_reports, tuned_csv_relative_to_reports)
_MODELS = [
    ("logistic_regression", "metrics_logistic_regression.csv",  "tuned/metrics_logistic_regression.csv"),
    ("naive_bayes",          "metrics_naive_bayes.csv",          "tuned/metrics_naive_bayes.csv"),
    ("random_forest",        "metrics_random_forest.csv",        "tuned/metrics_random_forest.csv"),
    ("xgboost",              "metrics_xgboost.csv",              "tuned/metrics_xgboost.csv"),
    ("lightgbm",             "metrics_lightgbm.csv",             "tuned/metrics_lightgbm.csv"),
    ("bilstm",               "metrics_bilstm.csv",               "tuned/metrics_bilstm.csv"),
    ("distilbert",           "metrics_distilbert.csv",           "tuned/metrics_distilbert.csv"),
    ("textgcn_improved",     "metrics_textgcn_improved.csv",     None),
    ("textgcn_tuned",        None,                               "tuned/metrics_textgcn_tuned.csv"),
]

def _load(rel_path: str) -> dict | None:
    try:
        df = pd.read_csv(PATHS.reports / rel_path)
        return df.iloc[0].to_dict() if not df.empty else None
    except FileNotFoundError:
        return None

def main() -> None:
    PATHS.reports.mkdir(parents=True, exist_ok=True)
    rows = []
    for display_name, base_file, tuned_file in _MODELS:
        for version, file in [("baseline", base_file), ("tuned", tuned_file)]:
            if file is None:
                continue
            row = _load(file)
            if row is None:
                print(f"  skipped (not found): {file}")
                continue
            row["model"]   = display_name
            row["version"] = version
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.drop(columns=[c for c in _DROP if c in df.columns], errors="ignore")
    df = df.sort_values(["model", "version"]).reset_index(drop=True)

    out = PATHS.reports / "metrics_model_comparison.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")

    summary = (
        df.sort_values("emscad_test_f1", ascending=False)
          [["model", "version", "emscad_test_f1", "emscad_test_precision",
            "emscad_test_recall", "openbay_recall"]]
          .round(4)
    )
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
