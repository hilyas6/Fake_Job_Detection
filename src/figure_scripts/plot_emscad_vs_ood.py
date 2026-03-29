# Scatter: EMSCAD F1 vs OpenBay Recall for all tuned models.
# The top-right corner is the ideal zone — high accuracy AND good generalisation.
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

FIGURES = Path(__file__).resolve().parents[2] / "reports" / "figures"
REPORTS = Path(__file__).resolve().parents[2] / "reports"

FAMILY = {
    "logistic_regression": ("#4C72B0", "Classical ML"),
    "naive_bayes":          ("#4C72B0", "Classical ML"),
    "random_forest":        ("#4C72B0", "Classical ML"),
    "xgboost":              ("#4C72B0", "Classical ML"),
    "lightgbm":             ("#4C72B0", "Classical ML"),
    "bilstm":               ("#DD8452", "Deep Learning"),
    "distilbert":           ("#DD8452", "Deep Learning"),
    "textgcn_tuned":        ("#C44E52", "TextGCN (Graph)"),
    "textgcn_improved":     ("#C44E52", "TextGCN (Graph)"),
}
SHORT = {
    "logistic_regression": "LogReg", "naive_bayes": "Naive Bayes",
    "random_forest": "Random\nForest", "xgboost": "XGBoost",
    "lightgbm": "LightGBM", "bilstm": "Bi-LSTM",
    "distilbert": "DistilBERT", "textgcn_tuned": "TextGCN\n(Tuned)",
    "textgcn_improved": "TextGCN",
}

def main():
    df = pd.read_csv(REPORTS / "metrics_model_comparison.csv")
    df = df[df["version"] == "tuned"].copy()

    fig, ax = plt.subplots(figsize=(9, 6))
    seen = set()
    for _, row in df.iterrows():
        m = row["model"]
        colour, family = FAMILY.get(m, ("#888", "Other"))
        lbl = family if family not in seen else None
        seen.add(family)
        size = 160 if m == "textgcn_tuned" else 100
        ax.scatter(row["emscad_test_f1"], row["openbay_recall"],
                   color=colour, s=size, zorder=3, label=lbl,
                   edgecolors="white", linewidth=0.8)
        ax.annotate(SHORT.get(m, m),
                    (row["emscad_test_f1"], row["openbay_recall"]),
                    textcoords="offset points", xytext=(6, 3), fontsize=8)

    # Ideal zone shading
    ax.axhspan(0.9, 1.02, alpha=0.06, color="#4CAF50")
    ax.axvspan(0.83, 0.88, alpha=0.06, color="#4CAF50")
    ax.text(0.855, 0.915, "Ideal zone", fontsize=8, color="#4CAF50", style="italic")

    ax.set_xlabel("EMSCAD Test F1  (In-Distribution Accuracy)", fontsize=11)
    ax.set_ylabel("OpenBay Recall  (Out-of-Distribution Generalisation)", fontsize=11)
    ax.set_title("In-Distribution Accuracy vs Out-of-Distribution Generalisation",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "emscad_vs_ood_scatter.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved → emscad_vs_ood_scatter.png")

if __name__ == "__main__":
    main()
