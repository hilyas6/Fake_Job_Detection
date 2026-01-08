# src/config.py  -- precision-focused setup

DATA_PATH = "data/fake_job_postings.csv"

# preprocessing
MIN_WORD_FREQ = 10          # prune rare words
MAX_TFIDF_FEATURES = 5000
SVD_DIM = 300

# graph
USE_PMI_EDGES = True
PMI_WINDOW = 20
PMI_MIN_COOCCUR = 10        # drop weak PMI links

# model
HIDDEN_DIM = 256
DROPOUT = 0.6

# training
LR = 0.003                  # smaller learning rate for stable training
WEIGHT_DECAY = 5e-4
EPOCHS = 300
EARLY_STOP_PATIENCE = 40
RANDOM_STATE = 42

# imbalance handling  →  use Focal Loss instead of class weights
USE_FOCAL_LOSS = True
ALPHA_POS = 0.55            # lower alpha → less fake bias
GAMMA = 2.0
REBALANCE_RATIO = 2         # keep 2:1 ratio real:fake in training only

# threshold tuning  →  aim for higher precision
TARGET_RECALL = 0.70

# output
MODEL_DIR = "outputs/models"
RESULT_DIR = "outputs/results"
