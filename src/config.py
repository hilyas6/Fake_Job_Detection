# src/config.py
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    root: Path = Path(".")
    data_raw: Path = Path("data/raw")
    data_processed: Path = Path("data/processed")
    models_baselines: Path = Path("models/baselines")
    models_textgcn: Path = Path("models/textgcn")
    models_comparison: Path = Path("models/comparison")
    reports: Path = Path("reports")
    figures: Path = Path("reports/figures")

@dataclass(frozen=True)
class Settings:
    seed: int = 42

    # EMSCAD split proportions
    test_size: float = 0.20   # final test share
    val_size: float = 0.10    # validation share (from full dataset)

    # Text building
    text_fields: tuple = ("title", "company_profile", "description", "requirements", "benefits")
    sep_token: str = " [SEP] "

    # TF-IDF
    tfidf_min_df: int = 2
    tfidf_max_df: float = 0.9
    tfidf_ngrams: tuple = (1, 2)

PATHS = Paths()
CFG = Settings()
