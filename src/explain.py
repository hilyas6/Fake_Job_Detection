# src/explain.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from src.config import PATHS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tfidf_lr", choices=["tfidf_lr"])
    args = ap.parse_args()

    PATHS.figures.mkdir(parents=True, exist_ok=True)

    em = pd.read_csv(PATHS.data_processed / "emscad.csv")
    vec = joblib.load(PATHS.models_baselines / "vectorizer.joblib")
    model = joblib.load(PATHS.models_baselines / f"{args.model}.joblib")

    # Use a sample as background for SHAP
    sample = em.sample(n=min(2000, len(em)), random_state=42)
    X = vec.transform(sample["text"])

    explainer = shap.LinearExplainer(model, X, feature_perturbation="interventional")
    shap_vals = explainer(X)

    # Global importance = mean(|shap|)
    mean_abs = np.mean(np.abs(shap_vals.values), axis=0)
    feat_names = vec.get_feature_names_out()

    topk = 25
    idx = np.argsort(mean_abs)[-topk:][::-1]
    top_feats = feat_names[idx]
    top_vals = mean_abs[idx]

    plt.figure()
    plt.barh(list(reversed(top_feats)), list(reversed(top_vals)))
    plt.title("Global SHAP importance (TF-IDF + Logistic Regression)")
    plt.tight_layout()

    out_path = PATHS.figures / "shap_global_tfidf_lr.png"
    plt.savefig(out_path, dpi=200)
    print("✅ Saved:", out_path)

if __name__ == "__main__":
    main()
