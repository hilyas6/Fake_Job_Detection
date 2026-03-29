# Grouped bar chart showing 3-fold CV F1 for TextGCN and LightGBM with mean annotated.
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURES = Path(__file__).resolve().parents[2] / "reports" / "figures"
REPORTS = Path(__file__).resolve().parents[2] / "reports"

COLOURS = {"TextGCN": "#C44E52", "LightGBM": "#4C72B0"}

def main():
    df = pd.read_csv(REPORTS / "cross_validation_results.csv")

    models = df["model"].unique()
    folds  = sorted(df["fold"].unique())
    x = np.arange(len(folds))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, model in enumerate(models):
        sub  = df[df["model"] == model].sort_values("fold")
        vals = sub["f1"].tolist()
        mean = np.mean(vals)
        std  = np.std(vals)
        offset = (i - 0.5) * w + w / 2
        bars = ax.bar(x + offset, vals, w, label=model,
                      color=COLOURS.get(model, "#888"), edgecolor="white", alpha=0.85)
        # Annotate mean ± std — alternate above/below to avoid overlap
        ax.axhline(mean, color=COLOURS.get(model, "#888"), linestyle="--", linewidth=1.2, alpha=0.6)
        va = "bottom" if i == 0 else "top"
        y_off = 0.004 if i == 0 else -0.004
        ax.text(len(folds) - 0.45, mean + y_off,
                f"Mean={mean:.3f}\n±{std:.3f}", fontsize=8,
                color=COLOURS.get(model, "#888"), fontweight="bold",
                va=va, ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in folds], fontsize=11)
    ax.set_ylim(0.70, 0.87)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title("3-Fold Cross-Validation: TextGCN vs LightGBM",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "cross_validation.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved → cross_validation.png")

if __name__ == "__main__":
    main()
