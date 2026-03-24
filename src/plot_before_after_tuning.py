"""Before vs After tuning comparison — horizontal bar chart for all model families."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PATHS

METRICS = ["emscad_test_f1", "emscad_test_precision", "emscad_test_recall"]
LABELS  = {"emscad_test_f1": "F1", "emscad_test_precision": "Precision", "emscad_test_recall": "Recall"}

# (display_name, before_csv, after_csv)  — all paths relative to reports/
_PAIRS = [
    ("BiLSTM",               "metrics_bilstm.csv",               "tuned/metrics_bilstm.csv"),
    ("DistilBERT",           "metrics_distilbert.csv",           "tuned/metrics_distilbert.csv"),
    ("LightGBM",             "metrics_lightgbm.csv",             "tuned/metrics_lightgbm.csv"),
    ("Logistic Regression",  "metrics_logistic_regression.csv",  "tuned/metrics_logistic_regression.csv"),
    ("Naive Bayes",          "metrics_naive_bayes.csv",          "tuned/metrics_naive_bayes.csv"),
    ("Random Forest",        "metrics_random_forest.csv",        "tuned/metrics_random_forest.csv"),
    ("TextGCN",              "metrics_textgcn_improved.csv",     "tuned/metrics_textgcn_tuned.csv"),
    ("XGBoost",              "metrics_xgboost.csv",              "tuned/metrics_xgboost.csv"),
]

COLOR_BEFORE = "#7f8c8d"
COLOR_AFTER  = "#1abc9c"


def _load(rel: str) -> pd.Series | None:
    try:
        df = pd.read_csv(PATHS.reports / rel)
        return df.iloc[0] if not df.empty else None
    except FileNotFoundError:
        return None


def main() -> None:
    names, before_rows, after_rows = [], [], []
    for name, before_path, after_path in _PAIRS:
        b = _load(before_path)
        a = _load(after_path)
        if b is None or a is None:
            print(f"  skipped {name} (missing file)")
            continue
        names.append(name)
        before_rows.append(b)
        after_rows.append(a)

    y = np.arange(len(names))
    bh = 0.36

    fig, axes = plt.subplots(1, 3, figsize=(20, max(6, len(names) * 0.8 + 2)), sharey=True)
    fig.patch.set_facecolor("white")

    all_vals = []
    for row in before_rows + after_rows:
        all_vals.extend([float(row[m]) for m in METRICS if m in row])
    x_min = max(0.0, min(all_vals) - 0.04)
    x_max = min(1.0, max(all_vals) + 0.025)

    for ax, metric in zip(axes, METRICS):
        b_vals = [float(r[metric]) if metric in r else 0.0 for r in before_rows]
        a_vals = [float(r[metric]) if metric in r else 0.0 for r in after_rows]

        bars_b = ax.barh(y - bh / 2, b_vals, height=bh, label="Before tuning", color=COLOR_BEFORE, edgecolor="white", linewidth=0.5)
        bars_a = ax.barh(y + bh / 2, a_vals, height=bh, label="After tuning",  color=COLOR_AFTER,  edgecolor="white", linewidth=0.5)

        # Annotate improvement deltas
        for i, (bv, av) in enumerate(zip(b_vals, a_vals)):
            delta = av - bv
            if abs(delta) > 0.001:
                color = "#27ae60" if delta > 0 else "#e74c3c"
                ax.text(max(av, bv) + 0.003, y[i] + bh / 2, f"{delta:+.3f}",
                        va="center", ha="left", fontsize=7.5, color=color, fontweight="bold")

        ax.set_title(LABELS[metric], fontweight="bold", fontsize=13, pad=10)
        ax.set_xlim(x_min, x_max - 0.01)
        ax.set_xlabel("Score", fontsize=10)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=10)
        ax.grid(axis="x", alpha=0.25, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].legend(loc="lower right", fontsize=9, framealpha=0.7)
    fig.suptitle("Model Performance: Before vs After Tuning", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()

    out = PATHS.reports / "figures" / "04_before_after_tuning_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
