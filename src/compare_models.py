# src/compare_models.py
import pandas as pd

from src.config import PATHS


def main():
    PATHS.reports.mkdir(parents=True, exist_ok=True)

    metric_files = [
        path
        for path in sorted(PATHS.reports.glob("metrics_*.csv"))
        if path.name
        not in {
            "metrics_baselines.csv",
            "metrics_classical_models.csv",
            "metrics_model_comparison.csv",
        }
    ]
    if not metric_files:
        raise FileNotFoundError("No metrics_*.csv files found in reports/.")

    rows = []
    for path in metric_files:
        df = pd.read_csv(path)
        if df.empty:
            continue
        rows.append(df.iloc[0])

    combined = pd.DataFrame(rows)
    combined = combined.sort_values("emscad_test_f1", ascending=False)
    out_path = PATHS.reports / "metrics_model_comparison.csv"
    combined.to_csv(out_path, index=False)
    print("✅ Saved:", out_path)
    print(combined)


if __name__ == "__main__":
    main()
