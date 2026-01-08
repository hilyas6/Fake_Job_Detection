# src/train_baselines.py
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from src.config import PATHS, CFG

def load_splits():
    with open(PATHS.data_processed / "splits.json", "r", encoding="utf-8") as f:
        return json.load(f)

def subset_by_ids(df: pd.DataFrame, ids):
    ids = set(map(str, ids))
    return df[df["id"].astype(str).isin(ids)].copy()

def main():
    PATHS.models_baselines.mkdir(parents=True, exist_ok=True)

    em = pd.read_csv(PATHS.data_processed / "emscad.csv")
    splits = load_splits()

    train = subset_by_ids(em, splits["train_ids"])
    val   = subset_by_ids(em, splits["val_ids"])
    # (We train on train only for now; you can later train train+val for final)
    X_train = train["text"]
    y_train = train["fraudulent"].astype(int)

    vec = TfidfVectorizer(
        ngram_range=CFG.tfidf_ngrams,
        min_df=CFG.tfidf_min_df,
        max_df=CFG.tfidf_max_df
    )
    Xtr = vec.fit_transform(X_train)

    models = {
        "tfidf_lr": LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1),
        "tfidf_svm": LinearSVC(class_weight="balanced"),
        "tfidf_nb": MultinomialNB(),
    }

    # Save vectorizer
    joblib.dump(vec, PATHS.models_baselines / "vectorizer.joblib")

    for name, model in models.items():
        model.fit(Xtr, y_train)
        joblib.dump(model, PATHS.models_baselines / f"{name}.joblib")
        print(f"✅ Saved model: models/baselines/{name}.joblib")

    print("✅ Done training baselines.")

if __name__ == "__main__":
    main()
