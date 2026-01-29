# src/model_utils.py
import json
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from src.config import PATHS, CFG

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str):
    if not isinstance(text, str):
        return []
    return TOKEN_RE.findall(text.lower())


def load_splits():
    with open(PATHS.data_processed / "splits.json", "r", encoding="utf-8") as f:
        return json.load(f)


def subset_by_ids(df: pd.DataFrame, ids):
    ids = set(map(str, ids))
    return df[df["id"].astype(str).isin(ids)].copy()


def eval_at_threshold(y_true, scores, threshold):
    pred = (scores >= threshold).astype(int)
    f1 = f1_score(y_true, pred, zero_division=0)
    prec = precision_score(y_true, pred, zero_division=0)
    rec = recall_score(y_true, pred, zero_division=0)
    return f1, prec, rec


def find_best_threshold(y_true, scores, grid=None):
    if grid is None:
        grid = np.linspace(0.1, 0.9, 41)
    best = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for t in grid:
        f1, prec, rec = eval_at_threshold(y_true, scores, float(t))
        if f1 > best["f1"] + 1e-6:
            best = {"threshold": float(t), "f1": f1, "precision": prec, "recall": rec}
    return best


def openbay_metrics(scores, threshold):
    pred = (scores >= threshold).astype(int)
    return {
        "openbay_recall": float(np.mean(pred == 1)),
        "openbay_mean_prob": float(np.mean(scores)),
        "openbay_median_prob": float(np.median(scores)),
    }


@dataclass
class TfidfConfig:
    ngram_range: tuple = (1, 3)
    min_df: int = CFG.tfidf_min_df
    max_df: float = CFG.tfidf_max_df
    max_features: int = 40000


def build_vocab(tokenized_docs, min_freq=2, max_size=40000):
    counter = Counter()
    for tokens in tokenized_docs:
        counter.update(tokens)

    vocab = {"<pad>": 0, "<unk>": 1}
    for token, count in counter.most_common():
        if count < min_freq:
            continue
        if len(vocab) >= max_size:
            break
        vocab[token] = len(vocab)
    return vocab


def encode_tokens(tokens, vocab):
    unk = vocab.get("<unk>", 1)
    return [vocab.get(tok, unk) for tok in tokens]
