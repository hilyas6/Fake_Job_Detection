import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from .config import RAW_CSV, TEXT_COLS, TARGET_COL, VECT_MAX_FEATURES, VECT_STOP_WORDS

def load_data(csv_path=RAW_CSV):
    df = pd.read_csv(csv_path)
    return df

def build_text_column(df):
    df = df.copy()
    df["text"] = df[TEXT_COLS].fillna("").agg(" ".join, axis=1)
    return df

def vectorize_text(train_text, test_text):
    vectorizer = TfidfVectorizer(stop_words=VECT_STOP_WORDS,
                                 max_features=VECT_MAX_FEATURES,
                                 lowercase=True)
    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)
    return X_train, X_test, vectorizer

def train_test(df, test_size=0.2, seed=42):
    X = df["text"]
    y = df[TARGET_COL]
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_train_text, X_test_text, y_train, y_test
