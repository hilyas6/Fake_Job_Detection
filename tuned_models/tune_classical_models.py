from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier

from tuned_models.common import (
    MODELS_DIR,
    REPORTS_DIR,
    ensure_dirs,
    eval_at_threshold,
    find_best_threshold,
    load_bundle,
    openbay_metrics,
    save_metrics_row,
)


def score_model(model, X_val, y_val):
    val_scores = model.predict_proba(X_val)[:, 1]
    return find_best_threshold(y_val, val_scores)


def main() -> None:
    ensure_dirs()
    bundle = load_bundle()

    vec = TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.92,
        max_features=70000,
        sublinear_tf=True,
    )

    X_train = vec.fit_transform(bundle.train_df["text"])
    X_val = vec.transform(bundle.val_df["text"])
    X_test = vec.transform(bundle.test_df["text"])
    X_ob = vec.transform(bundle.openbay_df["text"])

    y_train = bundle.train_df["fraudulent"].astype(int).values
    y_val = bundle.val_df["fraudulent"].astype(int).values
    y_test = bundle.test_df["fraudulent"].astype(int).values

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    spw = float(neg / max(pos, 1))

    joblib.dump(vec, MODELS_DIR / "vectorizer.joblib")

    candidates = {
        "logistic_regression": [
            LogisticRegression(C=c, class_weight="balanced", solver="saga", max_iter=3000, n_jobs=-1)
            for c in [0.5, 1.0, 2.0]
        ],
        "naive_bayes": [
            MultinomialNB(alpha=a)
            for a in [0.1, 0.25, 0.5, 1.0]
        ],
        "random_forest": [
            RandomForestClassifier(
                n_estimators=600,
                max_depth=depth,
                min_samples_leaf=leaf,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=42,
            )
            for depth in [None, 30]
            for leaf in [1, 2]
        ],
        "xgboost": [
            XGBClassifier(
                n_estimators=700,
                learning_rate=lr,
                max_depth=depth,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=spw,
                n_jobs=-1,
                random_state=42,
            )
            for lr in [0.03, 0.05]
            for depth in [5, 7]
        ],
        "lightgbm": [
            LGBMClassifier(
                n_estimators=800,
                learning_rate=lr,
                num_leaves=leaves,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="binary",
                scale_pos_weight=spw,
                n_jobs=-1,
                random_state=42,
            )
            for lr in [0.03, 0.05]
            for leaves in [63, 127]
        ],
    }

    summary = []

    for name, models in candidates.items():
        best = {"f1": -1.0}
        best_model = None
        for model in models:
            model.fit(X_train, y_train)
            val = score_model(model, X_val, y_val)
            if val["f1"] > best["f1"]:
                best = val
                best_model = model

        assert best_model is not None
        joblib.dump(best_model, MODELS_DIR / f"{name}.joblib")

        test_scores = best_model.predict_proba(X_test)[:, 1]
        test = eval_at_threshold(y_test, test_scores, best["threshold"])

        ob_scores = best_model.predict_proba(X_ob)[:, 1]
        ob = openbay_metrics(ob_scores, best["threshold"])

        metrics = {
            "emscad_test_f1": test["f1"],
            "emscad_test_precision": test["precision"],
            "emscad_test_recall": test["recall"],
            "threshold": best["threshold"],
            **ob,
        }
        save_metrics_row(name, metrics)
        summary.append({"model": name, **metrics})
        print(f"Saved tuned {name}")

    pd.DataFrame(summary).sort_values("emscad_test_f1", ascending=False).to_csv(
        REPORTS_DIR / "metrics_classical_models.csv", index=False
    )


if __name__ == "__main__":
    main()
