"""
Dataset Statistics
==================
Generates class distribution per split (Train / Val / Test) for the EMSCAD
dataset and a summary of the OpenBay out-of-distribution set.

Outputs
-------
  reports/dataset_statistics.csv   – machine-readable table
  prints a formatted summary to stdout
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS_DIR    = ROOT / "reports"


def main():
    em = pd.read_csv(DATA_PROCESSED / "emscad.csv")
    ob = pd.read_csv(DATA_PROCESSED / "openbay.csv")

    with open(DATA_PROCESSED / "splits.json", encoding="utf-8") as f:
        splits = json.load(f)

    def subset(df, ids):
        return df[df["id"].astype(str).isin(set(map(str, ids)))]

    train_df = subset(em, splits["train_ids"])
    val_df   = subset(em, splits["val_ids"])
    test_df  = subset(em, splits["test_ids"])

    rows = []
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df), ("EMSCAD Total", em)]:
        real  = int((df["fraudulent"] == 0).sum())
        fake  = int((df["fraudulent"] == 1).sum())
        total = real + fake
        rows.append({
            "split":    name,
            "real":     real,
            "fake":     fake,
            "total":    total,
            "fake_pct": round(fake / total * 100, 2) if total > 0 else 0.0,
        })

    # OpenBay has no ground-truth labels
    rows.append({
        "split":    "OpenBay (OOD)",
        "real":     "N/A",
        "fake":     "N/A",
        "total":    len(ob),
        "fake_pct": "unknown",
    })

    stats_df = pd.DataFrame(rows)
    out_path = REPORTS_DIR / "dataset_statistics.csv"
    stats_df.to_csv(out_path, index=False)

    # Pretty print
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"{'Split':<20} {'Real':>8} {'Fake':>8} {'Total':>8} {'Fake %':>8}")
    print("-" * 60)
    for _, row in stats_df.iterrows():
        fake_pct = f"{row['fake_pct']}%" if row["fake_pct"] != "unknown" else "unknown"
        print(f"{row['split']:<20} {str(row['real']):>8} {str(row['fake']):>8} {str(row['total']):>8} {fake_pct:>8}")
    print("=" * 60)
    print(f"\nSaved to: {out_path}")

    # Additional text-length stats
    print("\nText length stats (EMSCAD, by label):")
    for label, name in [(0, "Real"), (1, "Fake")]:
        sub = em[em["fraudulent"] == label]["text"]
        lengths = sub.str.len()
        print(f"  {name}: mean={lengths.mean():.0f} median={lengths.median():.0f} "
              f"min={lengths.min()} max={lengths.max()}")


if __name__ == "__main__":
    main()
