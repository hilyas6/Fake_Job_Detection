# src/preprocess.py
import re
import nltk
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np

nltk.download("punkt", quiet=True)

_bert_model = None
def get_bert_model():
    global _bert_model
    if _bert_model is None:
        _bert_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _bert_model


def clean_and_tokenize(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    return [t for t in tokens if len(t) > 2]


def add_bert_embeddings(df):
    model = get_bert_model()
    texts = df["text"].tolist()
    embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, batch_size=32)
    df["bert_emb"] = list(embs)
    return df


def add_handcrafted_features(df):
    """Add simple binary indicators of scammy traits."""
    scam_words = ["money", "wire", "fee", "gift card", "payment", "western union",
                  "bitcoin", "transfer", "cash", "deposit", "credit card"]
    df["has_email_free_domain"] = df["description"].str.contains(
        r"@gmail\.com|@yahoo\.com|@outlook\.com", case=False, na=False
    ).astype(int)
    df["mentions_payment"] = df["description"].str.contains(
        "|".join(scam_words), case=False, na=False
    ).astype(int)
    df["missing_company_profile"] = df["company_profile"].isna().astype(int)
    return df
