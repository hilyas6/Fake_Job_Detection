# src/dataset_loader.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import torch
from .preprocess import clean_and_tokenize, add_bert_embeddings, add_handcrafted_features
from . import config

def load_dataset(path):
    df = pd.read_csv(path)
    df["text"] = df["title"].fillna("") + " " + df["description"].fillna("") + " " + df["requirements"].fillna("")
    df["tokens"] = df["text"].apply(clean_and_tokenize)
    df = add_handcrafted_features(df)
    df = add_bert_embeddings(df)
    df["label"] = df["fraudulent"].astype(int)

    train_df, test_df = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=config.RANDOM_STATE)
    val_df, test_df = train_test_split(test_df, test_size=0.5, stratify=test_df["label"], random_state=config.RANDOM_STATE)
    return train_df, val_df, test_df


def build_doc_features(df):
    """Concatenate BERT + TF-IDF mean + heuristic binary features."""
    tfidf = TfidfVectorizer(max_features=config.MAX_TFIDF_FEATURES, min_df=config.MIN_WORD_FREQ, max_df=0.8)
    X_tfidf = tfidf.fit_transform(df["text"])
    X_tfidf_mean = np.array(X_tfidf.mean(axis=1)).reshape(-1, 1)
    X_bert = np.stack(df["bert_emb"].values)
    X_rules = df[["has_email_free_domain", "mentions_payment", "missing_company_profile"]].values
    X = np.hstack([X_bert, X_tfidf_mean, X_rules])
    return torch.tensor(X, dtype=torch.float32)
