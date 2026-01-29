# src/train_classical_models.py
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import PATHS
from src.model_utils import (
    TfidfConfig,
    eval_at_threshold,
    find_best_threshold,
    load_splits,
    openbay_metrics,
    subset_by_ids,
)


def build_vectorizer(cfg: TfidfConfig):
    return TfidfVectorizer(
        ngram_range=cfg.ngram_range,
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        sublinear_tf=True,
        max_features=cfg.max_features,
    )


def model_registry(scale_pos_weight: float):
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "naive_bayes": MultinomialNB(),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "xgboost": XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            random_state=42,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary",
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            random_state=42,
        ),
    }


def main():
    PATHS.models_comparison.mkdir(parents=True, exist_ok=True)
    PATHS.reports.mkdir(parents=True, exist_ok=True)

    em = pd.read_csv(PATHS.data_processed / "emscad.csv")
    ob = pd.read_csv(PATHS.data_processed / "openbay.csv")
    splits = load_splits()

    train_df = subset_by_ids(em, splits["train_ids"])
    val_df = subset_by_ids(em, splits["val_ids"])
    test_df = subset_by_ids(em, splits["test_ids"])

    cfg = TfidfConfig()
    vec = build_vectorizer(cfg)
    X_train = vec.fit_transform(train_df["text"])
    X_val = vec.transform(val_df["text"])
    X_test = vec.transform(test_df["text"])
    X_ob = vec.transform(ob["text"])

    y_train = train_df["fraudulent"].astype(int).values
    y_val = val_df["fraudulent"].astype(int).values
    y_test = test_df["fraudulent"].astype(int).values

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = float(neg / max(pos, 1))

    joblib.dump(vec, PATHS.models_comparison / "vectorizer.joblib")

    rows = []
    for name, model in model_registry(scale_pos_weight).items():
        model.fit(X_train, y_train)
        joblib.dump(model, PATHS.models_comparison / f"{name}.joblib")

        val_scores = model.predict_proba(X_val)[:, 1]
        best_threshold = find_best_threshold(y_val, val_scores)

        test_scores = model.predict_proba(X_test)[:, 1]
        test_f1, test_p, test_r = eval_at_threshold(
            y_test, test_scores, best_threshold["threshold"]
        )

        ob_scores = model.predict_proba(X_ob)[:, 1]
        ob_metrics = openbay_metrics(ob_scores, best_threshold["threshold"])

        metrics = {
            "model": name,
            "emscad_test_f1": float(test_f1),
            "emscad_test_precision": float(test_p),
            "emscad_test_recall": float(test_r),
            "openbay_recall": ob_metrics["openbay_recall"],
            "openbay_mean_prob": ob_metrics["openbay_mean_prob"],
            "openbay_median_prob": ob_metrics["openbay_median_prob"],
            "threshold": float(best_threshold["threshold"]),
        }
        pd.DataFrame([metrics]).to_csv(
            PATHS.reports / f"metrics_{name}.csv", index=False
        )
        rows.append(metrics)
        print(f"✅ Saved {name} + metrics.")

    pd.DataFrame(rows).to_csv(PATHS.reports / "metrics_classical_models.csv", index=False)
    print(f"✅ Saved summary to {PATHS.reports / 'metrics_classical_models.csv'}")


if __name__ == "__main__":
    main()
