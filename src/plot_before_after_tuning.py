from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PATHS


METRICS = ["emscad_test_f1", "emscad_test_precision", "emscad_test_recall"]
METRIC_LABELS = {
    "emscad_test_f1": "F1",
    "emscad_test_precision": "Precision",
    "emscad_test_recall": "Recall",
}


def _read_metrics_row(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "model" not in df.columns:
        raise ValueError(f"Expected 'model' column in {path}")
    return df[["model", *METRICS]].copy()


def build_before_after_dataframe() -> pd.DataFrame:
    reports = PATHS.reports

    base_files = {
        "logistic_regression": reports / "metrics_logistic_regression.csv",
        "naive_bayes": reports / "metrics_naive_bayes.csv",
        "random_forest": reports / "metrics_random_forest.csv",
        "xgboost": reports / "metrics_xgboost.csv",
        "lightgbm": reports / "metrics_lightgbm.csv",
        "distilbert": reports / "metrics_distilbert.csv",
        "bilstm": reports / "metrics_bilstm.csv",
    }

    tuned_files = {
        "logistic_regression": reports / "tuned" / "metrics_logistic_regression.csv",
        "naive_bayes": reports / "tuned" / "metrics_naive_bayes.csv",
        "random_forest": reports / "tuned" / "metrics_random_forest.csv",
        "xgboost": reports / "tuned" / "metrics_xgboost.csv",
        "lightgbm": reports / "tuned" / "metrics_lightgbm.csv",
        "distilbert": reports / "tuned" / "metrics_distilbert.csv",
        "bilstm": reports / "tuned" / "metrics_bilstm.csv",
    }

    rows = []

    for model_name, file_path in base_files.items():
        row = _read_metrics_row(file_path).iloc[0].to_dict()
        row["model"] = model_name
        row["version"] = "Before tuning"
        rows.append(row)

    for model_name, file_path in tuned_files.items():
        row = _read_metrics_row(file_path).iloc[0].to_dict()
        row["model"] = model_name
        row["version"] = "After tuning"
        rows.append(row)

    # User requested mapping: normal TextGCN = base, improved TextGCN = tuned.
    textgcn_base = _read_metrics_row(reports / "metrics_textgcn.csv").iloc[0].to_dict()
    textgcn_base["model"] = "textgcn"
    textgcn_base["version"] = "Before tuning"
    rows.append(textgcn_base)

    textgcn_tuned = _read_metrics_row(reports / "metrics_textgcn_improved.csv").iloc[0].to_dict()
    textgcn_tuned["model"] = "textgcn"
    textgcn_tuned["version"] = "After tuning"
    rows.append(textgcn_tuned)

    df = pd.DataFrame(rows)
    ordered_models = sorted(df["model"].unique())
    df["model"] = pd.Categorical(df["model"], categories=ordered_models, ordered=True)
    return df.sort_values(["model", "version"]).reset_index(drop=True)


def plot_before_after_comparison(df: pd.DataFrame, out_path: Path) -> None:
    models = list(df["model"].cat.categories)
    y = np.arange(len(models))
    bar_height = 0.36

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)

    for ax, metric_col in zip(axes, METRICS):
        before_scores = []
        after_scores = []
        for model in models:
            model_rows = df[df["model"] == model]
            before_scores.append(float(model_rows[model_rows["version"] == "Before tuning"][metric_col].iloc[0]))
            after_scores.append(float(model_rows[model_rows["version"] == "After tuning"][metric_col].iloc[0]))

        ax.barh(y - bar_height / 2, before_scores, height=bar_height, label="Before tuning", color="#7f8c8d")
        ax.barh(y + bar_height / 2, after_scores, height=bar_height, label="After tuning", color="#1abc9c")

        ax.set_title(METRIC_LABELS[metric_col], fontweight="bold")
        ax.set_xlim(0.5, 1.0)
        ax.set_xlabel("Score")
        ax.set_yticks(y)
        ax.set_yticklabels(models)
        ax.grid(axis="x", alpha=0.25)

    axes[0].set_ylabel("Model")
    axes[0].legend(loc="lower right", title="Version")
    fig.suptitle("Model Performance Comparison: Before vs After Tuning", fontsize=18, fontweight="bold")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = build_before_after_dataframe()
    output_path = PATHS.reports / "figures" / "04_before_after_tuning_comparison.png"
    plot_before_after_comparison(df, output_path)
    print(f"Saved chart to {output_path}")


if __name__ == "__main__":
    main()
