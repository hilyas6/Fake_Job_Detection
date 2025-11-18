from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV = DATA_DIR / "fake_job_postings.csv"

TEXT_COLS = ["title", "company_profile", "description", "requirements", "benefits"]
TARGET_COL = "fraudulent"

VECT_MAX_FEATURES = 20000
VECT_STOP_WORDS = "english"
