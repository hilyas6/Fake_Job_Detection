# src/thresholds.py
import numpy as np
from sklearn.metrics import precision_recall_curve

def choose_threshold_for_recall(y_true, y_score, min_recall=0.85):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    thresholds = np.r_[0.0, thresholds]  # align lengths
    mask = recall >= min_recall
    if not mask.any():
        f1 = (2 * precision * recall) / (precision + recall + 1e-9)
        idx = np.nanargmax(f1)
        return thresholds[idx], precision[idx], recall[idx]
    idx = np.argmax(precision[mask])
    idx_global = np.where(mask)[0][idx]
    return thresholds[idx_global], precision[idx_global], recall[idx_global]
