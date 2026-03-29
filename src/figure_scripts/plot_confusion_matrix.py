# Confusion matrix for the tuned TextGCN model on the EMSCAD test set.
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

FIGURES = Path(__file__).resolve().parents[2] / "reports" / "figures"

# Values from error_analysis_summary.txt
TN, FP, FN, TP = 3017, 15, 32, 111

def main():
    matrix = np.array([[TN, FP], [FN, TP]])
    labels = [["True Negative\n(Correct: Real)", "False Positive\n(Missed: Real as Fake)"],
              ["False Negative\n(Missed: Fake as Real)", "True Positive\n(Correct: Fake)"]]

    colours = [["#4C72B0", "#F4A582"], ["#F4A582", "#C44E52"]]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, 1-i), 1, 1, color=colours[i][j], alpha=0.75))
            ax.text(j + 0.5, 1.5 - i, str(matrix[i, j]),
                    ha="center", va="center", fontsize=22, fontweight="bold", color="white")
            ax.text(j + 0.5, 1.5 - i - 0.28, labels[i][j],
                    ha="center", va="center", fontsize=8, color="white", alpha=0.9)

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(["Predicted: Real", "Predicted: Fake"], fontsize=11)
    ax.set_yticklabels(["Actual: Fake", "Actual: Real"], fontsize=11)
    ax.set_title("Confusion Matrix: TextGCN Tuned (EMSCAD Test Set, threshold=0.48)",
                 fontsize=12, fontweight="bold", pad=14)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    plt.tight_layout()

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "confusion_matrix.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved → confusion_matrix.png")

if __name__ == "__main__":
    main()
