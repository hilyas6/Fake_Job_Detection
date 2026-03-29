# Class distribution of real vs fake job postings across Train / Val / Test splits.
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

FIGURES = Path(__file__).resolve().parents[2] / "reports" / "figures"
REPORTS = Path(__file__).resolve().parents[2] / "reports"

COLOURS = {"Real": "#4C72B0", "Fake": "#C44E52"}

def main():
    df = pd.read_csv(REPORTS / "dataset_statistics.csv")
    df = df[df["split"].isin(["Train", "Val", "Test"])]

    splits = df["split"].tolist()
    reals  = df["real"].tolist()
    fakes  = df["fake"].tolist()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_r = ax.bar(splits, reals, color=COLOURS["Real"], label="Real", width=0.5)
    bars_f = ax.bar(splits, fakes, bottom=reals, color=COLOURS["Fake"], label="Fake", width=0.5)

    # Annotate fake counts on top
    for bar, fake, real in zip(bars_f, fakes, reals):
        ax.text(bar.get_x() + bar.get_width() / 2, real + fake + 80,
                f"{fake} fake\n({fake/(fake+real)*100:.1f}%)",
                ha="center", va="bottom", fontsize=9, color=COLOURS["Fake"], fontweight="bold")

    ax.set_title("Class Distribution Across Dataset Splits", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Job Postings", fontsize=11)
    ax.set_xlabel("Split", fontsize=11)
    ax.legend(fontsize=10, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "class_distribution.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved → class_distribution.png")

if __name__ == "__main__":
    main()
