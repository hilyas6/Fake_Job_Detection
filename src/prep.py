# src/prep.py
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PATHS, CFG

EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.I)
URL_RE   = re.compile(r"(https?://\S+|www\.\S+)", re.I)
PHONE_RE = re.compile(r"(\+?\d[\d\-\s\(\)]{7,}\d)")
MONEY_RE = re.compile(r"(\$|£|€)\s?\d[\d,]*(\.\d+)?", re.I)

def clean_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s)
    s = s.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    s = EMAIL_RE.sub("__EMAIL__", s)
    s = URL_RE.sub("__URL__", s)
    s = PHONE_RE.sub("__PHONE__", s)
    s = MONEY_RE.sub("__MONEY__", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_text(row: pd.Series) -> str:
    parts = []
    for col in CFG.text_fields:
        parts.append(clean_text(row.get(col, "")))
    parts = [p for p in parts if p]
    return CFG.sep_token.join(parts)

def ensure_dirs():
    PATHS.data_processed.mkdir(parents=True, exist_ok=True)

def load_emscad() -> pd.DataFrame:
    p = PATHS.data_raw / "fake_job_postings.csv"
    df = pd.read_csv(p)

    # Make sure 'id' exists
    if "job_id" in df.columns and "id" not in df.columns:
        df = df.rename(columns={"job_id": "id"})
    if "id" not in df.columns:
        df["id"] = df.index.astype(str)
    df["id"] = df["id"].astype(str)

    df["dataset"] = "emscad"
    df["text"] = df.apply(build_text, axis=1)

    # Keep label as int
    if "fraudulent" not in df.columns:
        raise ValueError("EMSCAD must contain 'fraudulent' column.")
    df["fraudulent"] = df["fraudulent"].astype(int)

    # Deduplicate by text (helps leakage/noise)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return df

def load_openbay() -> pd.DataFrame:
    p = PATHS.data_raw / "Fake_Postings.csv"
    df = pd.read_csv(p)

    if "id" not in df.columns:
        df["id"] = df.index.astype(str)
    df["id"] = df["id"].astype(str)

    df["dataset"] = "openbay"
    df["text"] = df.apply(build_text, axis=1)

    # Keep label as int if present
    if "fraudulent" in df.columns:
        df["fraudulent"] = df["fraudulent"].astype(int)

    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return df

def make_splits(emscad_df: pd.DataFrame) -> dict:
    # Split EMSCAD into train/val/test with stratification
    y = emscad_df["fraudulent"]

    # First carve out test set
    train_val, test = train_test_split(
        emscad_df, test_size=CFG.test_size, random_state=CFG.seed, stratify=y
    )

    # Then carve validation from remaining (val_size relative to full dataset)
    val_rel = CFG.val_size / (1.0 - CFG.test_size)  # fraction of train_val
    train, val = train_test_split(
        train_val, test_size=val_rel, random_state=CFG.seed, stratify=train_val["fraudulent"]
    )

    return {
        "seed": CFG.seed,
        "train_ids": train["id"].tolist(),
        "val_ids": val["id"].tolist(),
        "test_ids": test["id"].tolist(),
    }

def main():
    ensure_dirs()

    em = load_emscad()
    ob = load_openbay()

    # Save processed datasets
    em_out = PATHS.data_processed / "emscad.csv"
    ob_out = PATHS.data_processed / "openbay.csv"
    em.to_csv(em_out, index=False)
    ob.to_csv(ob_out, index=False)

    splits = make_splits(em)
    with open(PATHS.data_processed / "splits.json", "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    print("✅ Saved:")
    print(f" - {em_out}")
    print(f" - {ob_out}")
    print(f" - {PATHS.data_processed / 'splits.json'}")
    print("\nEMSCAD class counts:")
    print(em["fraudulent"].value_counts())
    if "fraudulent" in ob.columns:
        print("\nOpenDataBay class counts:")
        print(ob["fraudulent"].value_counts())
    else:
        print("\nOpenDataBay has no 'fraudulent' column (ok if unlabeled).")

if __name__ == "__main__":
    main()
