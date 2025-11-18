import pandas as pd
from src import load_data, build_text_column, train_test, vectorize_text
from src import PROJECT_ROOT


def main():
    df = load_data()
    df = build_text_column(df)
    X_train_text, X_test_text, y_train, y_test = train_test(df)
    X_train, X_test, _ = vectorize_text(X_train_text, X_test_text)

    results = run_baselines(X_train, y_train, X_test, y_test)

    print("\n==== Baseline Results ====\n")
    rows = []
    for model_name, m in results.items():
        print(f"## {model_name}")
        print(f"Accuracy: {m['accuracy']:.4f}")
        print(f"Precision: {m['precision']:.4f}")
        print(f"Recall:    {m['recall']:.4f}")
        print(f"F1:        {m['f1']:.4f}")
        print("Confusion Matrix:\n", m["confusion_matrix"])
        print("-" * 50)

        rows.append({"model": model_name, **m})

    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(artifacts_dir / "baseline_summary.csv", index=False)

if __name__ == "__main__":
    main()
