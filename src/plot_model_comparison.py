"""Generate comparison bar charts and heatmap from metrics_model_comparison.csv."""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import PATHS

_EMSCAD  = ["emscad_test_f1", "emscad_test_precision", "emscad_test_recall"]
_OPENBAY = ["openbay_recall", "openbay_mean_prob", "openbay_median_prob"]
_DROP    = {"vocab_size", "pmi_window_size", "pmi_threshold", "residual_alpha", "threshold", "trial", "version"}

def _pretty(col: str) -> str:
    s = col.replace("emscad_test_", "").replace("openbay_", "").replace("_", " ")
    return re.sub(r"\bf1\b", "F1", s, flags=re.I).title()

def _label(row: pd.Series) -> str:
    model = str(row["model"]).replace("_", " ").title()
    ver   = str(row.get("version", ""))
    return f"{model} (tuned)" if ver == "tuned" else model

def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {path}")


def plot_grouped(df: pd.DataFrame, metrics: list[str], title: str, out: Path) -> None:
    df2 = df.copy()
    df2["Model"] = df2.apply(_label, axis=1)
    df2 = df2.sort_values("emscad_test_f1", ascending=False)

    melted = df2.melt(id_vars=["Model"], value_vars=metrics, var_name="Metric", value_name="Score")
    melted["Metric"] = melted["Metric"].apply(_pretty)

    fig_h = max(6, len(df2) * 0.55 + 2)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    fig.patch.set_facecolor("white")

    sns.barplot(data=melted, y="Model", x="Score", hue="Metric",
                palette="viridis", edgecolor=".2", ax=ax)
    ax.set_title(title, fontweight="bold", fontsize=15, loc="left", pad=14)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Score (0–1)", fontsize=11)
    ax.set_ylabel("")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False, title="Metric")
    plt.tight_layout()
    _save(fig, out)


def plot_heatmap(df: pd.DataFrame, out: Path) -> None:
    df2 = df.copy()
    df2["Model"] = df2.apply(_label, axis=1)
    df2 = df2.sort_values("emscad_test_f1", ascending=False)

    numeric = [c for c in df2.columns if pd.api.types.is_numeric_dtype(df2[c]) and c not in _DROP]
    heat = df2.set_index("Model")[numeric]
    heat.columns = [_pretty(c) for c in heat.columns]

    fig, ax = plt.subplots(figsize=(14, max(6, len(df2) * 0.65 + 2)))
    fig.patch.set_facecolor("white")
    sns.heatmap(heat, annot=True, fmt=".3f", cmap="RdYlGn", linewidths=1.0,
                cbar_kws={"label": "Score"}, annot_kws={"size": 9, "weight": "bold"}, ax=ax)
    ax.set_title("All Models — Full Performance Matrix\n(sorted by EMSCAD Test F1)",
                 fontweight="bold", fontsize=14, loc="left", pad=16)
    plt.xticks(rotation=35, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    _save(fig, out)


def main() -> None:
    in_path = PATHS.reports / "metrics_model_comparison.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Run src.compare_models first — {in_path} not found.")

    df = pd.read_csv(in_path)
    out_dir = PATHS.reports / "figures"

    baseline = df[df["version"] == "baseline"] if "version" in df.columns else df

    plot_grouped(baseline, _EMSCAD,  "EMSCAD Test Results — Baseline Models", out_dir / "01_emscad_bars.png")
    plot_grouped(baseline, _OPENBAY, "OpenBay Generalisation — Baseline Models", out_dir / "02_openbay_bars.png")
    plot_heatmap(baseline, out_dir / "03_professional_heatmap.png")
    print("All figures saved.")


if __name__ == "__main__":
    main()
