from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models" / "tuned"
REPORTS_DIR = ROOT / "reports" / "tuned"


@dataclass
class DatasetBundle:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    openbay_df: pd.DataFrame


def ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def subset_by_ids(df: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    ids_set = set(map(str, ids))
    return df[df["id"].astype(str).isin(ids_set)].copy()


def load_bundle() -> DatasetBundle:
    em = pd.read_csv(DATA_PROCESSED / "emscad.csv")
    ob = pd.read_csv(DATA_PROCESSED / "openbay.csv")
    with open(DATA_PROCESSED / "splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)

    return DatasetBundle(
        train_df=subset_by_ids(em, splits["train_ids"]),
        val_df=subset_by_ids(em, splits["val_ids"]),
        test_df=subset_by_ids(em, splits["test_ids"]),
        openbay_df=ob,
    )


def eval_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(int)
    return {
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
    }


def find_best_threshold(y_true: np.ndarray, scores: np.ndarray) -> dict:
    grid = np.linspace(0.1, 0.9, 41)
    best = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for t in grid:
        metrics = eval_at_threshold(y_true, scores, float(t))
        if metrics["f1"] > best["f1"] + 1e-6:
            best = {"threshold": float(t), **metrics}
    return best


def openbay_metrics(scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(int)
    return {
        "openbay_recall": float(np.mean(pred == 1)),
        "openbay_mean_prob": float(np.mean(scores)),
        "openbay_median_prob": float(np.median(scores)),
    }


def save_metrics_row(name: str, metrics: dict) -> None:
    row = {"model": name, **metrics}
    pd.DataFrame([row]).to_csv(REPORTS_DIR / f"metrics_{name}.csv", index=False)
