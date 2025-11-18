# src/evaluate.py
import torch
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt


def evaluate_model(model, data, num_docs, result_dir, target_recall=None, fixed_threshold=0.5):
    """
    Evaluate TextGCN model on test (document) nodes.
    Supports either a fixed threshold or threshold tuning by recall.
    """
    model.eval()
    device = next(model.parameters()).device

    # Forward pass
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_weight)
        probs = torch.softmax(out[:num_docs], dim=1)[:, 1].cpu().numpy()

    # Ground truth labels
    y_true = data.y[:num_docs].cpu().numpy()

    # --- Threshold selection ---
    if target_recall is not None:
        prec, rec, thresh = precision_recall_curve(y_true, probs)
        best_idx = np.argmin(np.abs(rec - target_recall))
        best_thresh = thresh[best_idx]
        print(f"🎯 Tuned threshold for recall={target_recall:.2f}: {best_thresh:.3f} "
              f"(precision={prec[best_idx]:.3f}, recall={rec[best_idx]:.3f})")
    else:
        best_thresh = fixed_threshold
        print(f"🔧 Using fixed threshold = {best_thresh:.3f}")

    preds = (probs >= best_thresh).astype(int)

    # --- Evaluation report ---
    report = classification_report(y_true, preds, digits=4, target_names=["Real", "Fake"])
    cm = confusion_matrix(y_true, preds)

    print("\n--- Test Results ---")
    print(report)
    print("Confusion Matrix:\n", cm)

    # --- Save results ---
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, "report.txt"), "w") as f:
        f.write("--- Test Results ---\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm))

    # --- Optional: plot Precision–Recall curve ---
    prec, rec, _ = precision_recall_curve(y_true, probs)
    plt.figure()
    plt.plot(rec, prec, label="PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (Fake Job Detection)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, "precision_recall_curve.png"))
    plt.close()

    print(f"✅ Results & plots saved to {result_dir}")
