# Grouped bar chart: top-K SHAP features vs random-K features probability drop (fake samples).
# A larger drop from top-K confirms that SHAP explanations reflect real model decisions.
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURES = Path(__file__).resolve().parents[2] / "reports" / "figures"
REPORTS = Path(__file__).resolve().parents[2] / "reports"

K_VALUES = [1, 3, 5, 10]

def main():
    df = pd.read_csv(REPORTS / "faithfulness_results.csv")
    fake = df[df["fraudulent"] == 1]  # Focus on fake samples for clearest signal

    top_drops  = [fake[f"top_{k}_drop"].mean()  for k in K_VALUES]
    rand_drops = [fake[f"rand_{k}_drop"].mean() for k in K_VALUES]

    x = np.arange(len(K_VALUES))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w/2, top_drops,  w, label="Top-K SHAP Features Removed",
                color="#C44E52", edgecolor="white")
    b2 = ax.bar(x + w/2, rand_drops, w, label="Random-K Features Removed",
                color="#9DB2BF", edgecolor="white")

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.0003,
                f"{h:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"K = {k}" for k in K_VALUES], fontsize=11)
    ax.set_ylabel("Mean Drop in Fake Probability", fontsize=11)
    ax.set_title("SHAP Explanation Faithfulness: Fake Job Samples\n"
                 "Removing top SHAP features causes a larger prediction change than removing random features",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(fontsize=10, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "faithfulness_evaluation.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved → faithfulness_evaluation.png")

if __name__ == "__main__":
    main()
