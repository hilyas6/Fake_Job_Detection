# Scatter of 36 TextGCN tuning trials: EMSCAD F1 vs OpenBay Recall.
# Points are coloured by ensemble size; the winning trial is highlighted.
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

FIGURES = Path(__file__).resolve().parents[2] / "reports" / "figures"
REPORTS = Path(__file__).resolve().parents[2] / "reports"

ENSEMBLE_COLOURS = {1.0: "#9DB2BF", 3.0: "#4C72B0", 5.0: "#C44E52"}
BEST_TRIAL = "v2_ensemble3_window15"

def main():
    df = pd.read_csv(REPORTS / "tuned" / "textgcn_tuning_results.csv")

    # Drop SWA trials (they collapsed — outliers that distort the axes)
    df = df[df["emscad_test_f1"] > 0.5].copy()

    fig, ax = plt.subplots(figsize=(9, 6))

    for n_ens, group in df.groupby("n_ensemble"):
        colour = ENSEMBLE_COLOURS.get(float(n_ens), "#aaa")
        ax.scatter(group["emscad_test_f1"], group["openbay_recall"],
                   color=colour, s=70, alpha=0.75, edgecolors="white",
                   linewidth=0.5, label=f"{int(n_ens)}-seed ensemble", zorder=2)

    # Highlight best trial
    best = df[df["trial"] == BEST_TRIAL]
    if not best.empty:
        ax.scatter(best["emscad_test_f1"], best["openbay_recall"],
                   color="#F18F01", s=200, zorder=4, marker="*",
                   edgecolors="black", linewidth=0.5, label="Best trial")
        ax.annotate("Best\n(window=15, 3-seed)",
                    (best["emscad_test_f1"].values[0], best["openbay_recall"].values[0]),
                    textcoords="offset points", xytext=(8, -18),
                    fontsize=8, color="#F18F01", fontweight="bold")

    ax.set_xlabel("EMSCAD Test F1  (In-Distribution)", fontsize=11)
    ax.set_ylabel("OpenBay Recall  (Out-of-Distribution)", fontsize=11)
    ax.set_title("TextGCN Hyperparameter Search: 36 Trials\nColour = Ensemble Size",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "hyperparameter_search.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved → hyperparameter_search.png")

if __name__ == "__main__":
    main()
