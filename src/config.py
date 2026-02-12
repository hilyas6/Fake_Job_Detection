"""Project configuration.

This module exposes both legacy constants and structured `PATHS` / `CFG`
objects used across training and tuning scripts.
"""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT
    data_raw: Path = ROOT / "data" / "raw"
    data_processed: Path = ROOT / "data" / "processed"
    models_baselines: Path = ROOT / "models" / "baselines"
    models_textgcn: Path = ROOT / "models" / "textgcn"
    models_comparison: Path = ROOT / "models" / "comparison"
    reports: Path = ROOT / "reports"
    figures: Path = ROOT / "reports" / "figures"


@dataclass(frozen=True)
class Config:
    seed: int = 42
    test_size: float = 0.20
    val_size: float = 0.20
    text_fields: tuple[str, ...] = (
        "title",
        "company_profile",
        "description",
        "requirements",
        "benefits",
        "industry",
        "function",
        "location",
    )
    sep_token: str = " [SEP] "

    tfidf_ngrams: tuple[int, int] = (1, 2)
    tfidf_min_df: int = 2
    tfidf_max_df: float = 0.95


PATHS = Paths()
CFG = Config()

# Legacy constants kept for compatibility with older scripts.
DATA_PATH = str(PATHS.data_raw / "fake_job_postings.csv")
MIN_WORD_FREQ = 10
MAX_TFIDF_FEATURES = 5000
SVD_DIM = 300
USE_PMI_EDGES = True
PMI_WINDOW = 20
PMI_MIN_COOCCUR = 10
HIDDEN_DIM = 256
DROPOUT = 0.6
LR = 0.003
WEIGHT_DECAY = 5e-4
EPOCHS = 300
EARLY_STOP_PATIENCE = 40
RANDOM_STATE = CFG.seed
USE_FOCAL_LOSS = True
ALPHA_POS = 0.55
GAMMA = 2.0
REBALANCE_RATIO = 2
TARGET_RECALL = 0.70
MODEL_DIR = str(ROOT / "outputs" / "models")
RESULT_DIR = str(ROOT / "outputs" / "results")
