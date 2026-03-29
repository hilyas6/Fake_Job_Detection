# Grouped bar chart comparing 6 TextGCN architecture variants on F1 and OOD Recall.
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURES = Path(__file__).resolve().parents[2] / "reports" / "figures"
REPORTS = Path(__file__).resolve().parents[2] / "reports"

VARIANT_LABELS = {
    "full_3layer_w15":       "3-Layer + Residual\n(Reference)",
    "no_residual_3layer_w15":"3-Layer\nNo Residual",
    "1layer_w15":            "1-Layer GCN",
    "2layer_w15":            "2-Layer GCN",
    "random_edges_3layer":   "3-Layer\nRandom Edges",
    "window5_3layer":        "3-Layer\nWindow=5",
}

def main():
    df = pd.read_csv(REPORTS / "ablation_results.csv")
    df["label"] = df["variant"].map(VARIANT_LABELS).fillna(df["variant"])

    x = np.arange(len(df))
    w = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x - w/2, df["emscad_test_f1"],  w, label="EMSCAD Test F1",  color="#4C72B0", edgecolor="white")
    b2 = ax.bar(x + w/2, df["openbay_recall"],   w, label="OpenBay Recall", color="#C44E52", edgecolor="white")

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], fontsize=9)
    ax.set_ylim(0.3, 1.05)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("TextGCN Ablation Study: Architecture Variant Comparison",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "ablation_study.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved → ablation_study.png")

if __name__ == "__main__":
    main()
