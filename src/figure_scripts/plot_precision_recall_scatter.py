# Scatter plot of precision vs recall for all tuned models, coloured by model family.
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import pandas as pd

FIGURES = Path(__file__).resolve().parents[2] / "reports" / "figures"
REPORTS = Path(__file__).resolve().parents[2] / "reports"

FAMILY = {
    "logistic_regression": ("Classical ML",    "#4C72B0"),
    "naive_bayes":          ("Classical ML",    "#4C72B0"),
    "random_forest":        ("Classical ML",    "#4C72B0"),
    "xgboost":              ("Classical ML",    "#4C72B0"),
    "lightgbm":             ("Classical ML",    "#4C72B0"),
    "bilstm":               ("Deep Learning",   "#DD8452"),
    "distilbert":           ("Deep Learning",   "#DD8452"),
    "textgcn_tuned":        ("TextGCN (Graph)", "#C44E52"),
    "textgcn_improved":     ("TextGCN (Graph)", "#C44E52"),
}
LABELS = {
    "logistic_regression": "Logistic Regression",
    "naive_bayes":         "Naive Bayes",
    "random_forest":       "Random Forest",
    "xgboost":             "XGBoost",
    "lightgbm":            "LightGBM",
    "bilstm":              "Bi-LSTM",
    "distilbert":          "DistilBERT",
    "textgcn_tuned":       "TextGCN (Tuned)",
    "textgcn_improved":    "TextGCN (Improved)",
}
MARKERS = {
    "logistic_regression": "o",
    "naive_bayes":         "s",
    "random_forest":       "^",
    "xgboost":             "D",
    "lightgbm":            "P",
    "bilstm":              "X",
    "distilbert":          "*",
    "textgcn_tuned":       "h",
    "textgcn_improved":    "p",
}


def main():
    df = pd.read_csv(REPORTS / "metrics_model_comparison.csv")
    df = df[df["version"] == "tuned"].copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    shape_handles = []
    for _, row in df.iterrows():
        m = row["model"]
        _, colour = FAMILY.get(m, ("Other", "#888"))
        marker = MARKERS.get(m, "o")
        ax.scatter(row["emscad_test_precision"], row["emscad_test_recall"],
                   color=colour, marker=marker, s=180, zorder=3,
                   edgecolors="white", linewidth=1.0)
        shape_handles.append(
            mlines.Line2D([], [], color=colour, marker=marker, linestyle="None",
                          markersize=9, label=LABELS.get(m, m))
        )

    # Zoom into the actual data region with a bit of padding
    ax.set_xlim(0.58, 1.02)
    ax.set_ylim(0.68, 0.89)

    ax.set_xlabel("Precision", fontsize=12)
    ax.set_ylabel("Recall", fontsize=12)
    ax.set_title("Precision vs Recall: All Tuned Models (EMSCAD Test Set)",
                 fontsize=13, fontweight="bold", pad=12)

    # Legend 1 — shape per model
    leg1 = ax.legend(handles=shape_handles, fontsize=8.5, frameon=True,
                     title="Model", title_fontsize=9,
                     loc="upper left", bbox_to_anchor=(1.01, 1.0),
                     borderaxespad=0)
    ax.add_artist(leg1)

    # Legend 2 — colour per family, positioned just below leg1
    fig.canvas.draw()
    leg1_bottom = ax.transAxes.inverted().transform(
        leg1.get_window_extent(fig.canvas.get_renderer())
    )[0][1]

    colour_handles = [
        mpatches.Patch(color="#4C72B0", label="Classical ML"),
        mpatches.Patch(color="#DD8452", label="Deep Learning"),
        mpatches.Patch(color="#C44E52", label="TextGCN (Graph)"),
    ]
    ax.legend(handles=colour_handles, fontsize=8.5, frameon=True,
              title="Model Family", title_fontsize=9,
              loc="upper left", bbox_to_anchor=(1.01, leg1_bottom - 0.03),
              borderaxespad=0)

    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "precision_recall_scatter.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved → precision_recall_scatter.png")


if __name__ == "__main__":
    main()
