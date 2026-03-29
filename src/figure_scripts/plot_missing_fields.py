# Heatmap showing how often each job field is empty for real vs fake postings.
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FIGURES = Path(__file__).resolve().parents[2] / "reports" / "figures"
DATA    = Path(__file__).resolve().parents[2] / "data" / "processed" / "emscad.csv"

FIELDS = ["company_profile", "description", "requirements",
          "benefits", "salary_range", "department", "location", "title"]

FIELD_LABELS = {
    "company_profile": "Company Profile", "description": "Job Description",
    "requirements": "Requirements",      "benefits": "Benefits",
    "salary_range": "Salary Range",      "department": "Department",
    "location": "Location",              "title": "Job Title",
}

def main():
    df = pd.read_csv(DATA, usecols=FIELDS + ["fraudulent"])
    df["Label"] = df["fraudulent"].map({0: "Real", 1: "Fake"})

    # % of postings where the field is empty, per label
    records = {}
    for label in ["Real", "Fake"]:
        sub = df[df["Label"] == label]
        records[label] = {
            FIELD_LABELS[f]: round(sub[f].isnull().mean() * 100, 1)
            for f in FIELDS
        }

    heat = pd.DataFrame(records).T   # rows = Real/Fake, cols = fields

    fig, ax = plt.subplots(figsize=(11, 3.5))
    sns.heatmap(heat, annot=True, fmt=".1f", cmap="RdYlGn_r",
                linewidths=1.0, ax=ax,
                cbar_kws={"label": "% Empty"},
                annot_kws={"size": 10, "weight": "bold"})
    ax.set_title("Percentage of Empty Fields: Real vs Fake Job Postings",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
    plt.tight_layout()

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "missing_fields_heatmap.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved → missing_fields_heatmap.png")

if __name__ == "__main__":
    main()
