# Horizontal bar chart showing F1 improvement after tuning for each model.
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

FIGURES = Path(__file__).resolve().parents[2] / "reports" / "figures"
REPORTS = Path(__file__).resolve().parents[2] / "reports"

MODEL_NAMES = {
    "bilstm": "Bi-LSTM", "distilbert": "DistilBERT",
    "lightgbm": "LightGBM", "logistic_regression": "Logistic Regression",
    "naive_bayes": "Naive Bayes", "random_forest": "Random Forest",
    "textgcn_tuned": "TextGCN", "textgcn_improved": "TextGCN",
    "xgboost": "XGBoost",
}

def main():
    df = pd.read_csv(REPORTS / "metrics_model_comparison.csv")
    base  = df[df["version"] == "baseline"].copy()
    tuned = df[df["version"] == "tuned"].copy()

    # Match models by name, compute delta
    merged = base[["model", "emscad_test_f1"]].merge(
        tuned[["model", "emscad_test_f1"]], on="model", suffixes=("_base", "_tuned")
    )
    merged["delta"] = merged["emscad_test_f1_tuned"] - merged["emscad_test_f1_base"]
    merged["Model"] = merged["model"].map(MODEL_NAMES).fillna(merged["model"])
    merged = merged.sort_values("delta")

    colours = ["#C44E52" if d < 0 else "#4C72B0" for d in merged["delta"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(merged["Model"], merged["delta"], color=colours, height=0.55, edgecolor="white")

    for bar, val in zip(bars, merged["delta"]):
        ax.text(val + (0.002 if val >= 0 else -0.002),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center",
                ha="left" if val >= 0 else "right",
                fontsize=9, fontweight="bold",
                color="#4C72B0" if val >= 0 else "#C44E52")

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("F1 Score Improvement After Hyperparameter Tuning",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Change in F1 Score (Tuned − Baseline)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "tuning_improvement.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved → tuning_improvement.png")

if __name__ == "__main__":
    main()
