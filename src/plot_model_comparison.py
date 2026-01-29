import re
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import PATHS

# Columns to ignore for plotting
EXCLUDE_COLS = {"vocab_size", "pmi_window_size", "pmi_threshold", "residual_alpha", "threshold"}


# ---------- label helpers ----------
def prettify_metric_name(col: str) -> str:
    s = col.strip()
    s = s.replace("emscad_test_", "").replace("openbay_", "")
    s = s.replace("_", " ")
    s = re.sub(r"\bf1\b", "F1", s, flags=re.I)
    return s.title()


def pick_model_col(df: pd.DataFrame) -> str:
    candidates = ["model", "model_name", "name", "run_name"]
    for c in candidates:
        if c in df.columns: return c
    return df.columns[0]


def save_fig(fig, out_base: Path):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    # PNG only
    fig.savefig(str(out_base) + ".png", dpi=300, bbox_inches="tight", facecolor='white')


# ---------- plot functions ----------

def plot_grouped_comparison(df, model_col, metric_cols, title, out_dir, filename):
    """
    Switched to horizontal bars to ensure model names are perfectly readable.
    """
    df_melted = df.melt(id_vars=[model_col], value_vars=metric_cols,
                        var_name="Metric", value_name="Score")
    df_melted["Metric"] = df_melted["Metric"].apply(prettify_metric_name)

    # Increase height dynamically based on number of models
    fig_height = max(6, len(df[model_col].unique()) * 0.6)
    plt.figure(figsize=(12, fig_height))

    sns.set_context("talk", font_scale=0.7)
    sns.set_style("whitegrid", {'axes.spines.right': False, 'axes.spines.top': False})

    # Horizontal bar plot (y=model_col) avoids overlapping labels
    ax = sns.barplot(
        data=df_melted,
        y=model_col,
        x="Score",
        hue="Metric",
        palette="viridis",
        edgecolor=".2"
    )

    ax.set_title(title, loc='left', fontweight='bold', pad=20, fontsize=16)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Score (0.0 - 1.0)")
    ax.set_ylabel("")  # Model names are clear on the Y axis

    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False, title="Metric")
    plt.tight_layout()
    save_fig(plt.gcf(), out_dir / filename)
    plt.close()


def heatmap_all_metrics(df, model_col, metric_cols, sort_by, out_dir: Path):
    df2 = df.sort_values(sort_by, ascending=False).copy()
    plot_data = df2.set_index(model_col)[metric_cols]
    plot_data.columns = [prettify_metric_name(c) for c in plot_data.columns]

    plt.figure(figsize=(14, 8))
    sns.set_context("paper", font_scale=1.2)

    # RdYlGn for Red-Yellow-Green stoplight style
    ax = sns.heatmap(
        plot_data,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        linewidths=1.2,
        cbar_kws={'label': 'Performance Score'},
        annot_kws={"size": 10, "weight": "bold"}
    )

    ax.set_title(f"Performance Matrix: All Models\nSorted by {prettify_metric_name(sort_by)}",
                 loc='left', fontweight='bold', pad=25, fontsize=18)
    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    save_fig(plt.gcf(), out_dir / "03_professional_heatmap")
    plt.close()


# ---------- Main ----------

def main():
    reports = PATHS.reports
    in_path = reports / "metrics_model_comparison.csv"

    if not in_path.exists():
        return print(f"File not found: {in_path}")

    df = pd.read_csv(in_path)
    model_col = pick_model_col(df)

    emscad_metrics = ["emscad_test_f1", "emscad_test_precision", "emscad_test_recall"]
    openbay_metrics = ["openbay_recall", "openbay_mean_prob", "openbay_median_prob"]

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    all_metrics = [c for c in numeric_cols if c not in EXCLUDE_COLS and c != model_col]

    # Keep a consistent sort
    df = df.sort_values("emscad_test_f1", ascending=False)
    out_dir = reports / "figures"

    # 1. Horizontal Bars (Fixed overlapping labels)
    plot_grouped_comparison(df, model_col, emscad_metrics, "EMSCAD Classification Results", out_dir, "01_emscad_bars")
    plot_grouped_comparison(df, model_col, openbay_metrics, "Openbay Generalization Results", out_dir,
                            "02_openbay_bars")

    # 2. Heatmap
    if all_metrics:
        heatmap_all_metrics(df, model_col, all_metrics, "emscad_test_f1", out_dir)

    print(f"✅ Success: 2 horizontal bar charts and 1 heatmap saved to: {out_dir}")


if __name__ == "__main__":
    main()