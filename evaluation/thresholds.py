"""
Threshold selection methods for anomaly detection
Implements various threshold optimization strategies
"""
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, confusion_matrix

def select_threshold(mode, y_true, errors, manual_value=None):
    """
    Select optimal threshold using different strategies
    
    Args:
        mode: Threshold selection method ('f1', 'youden', 'avg')
        y_true: True labels
        errors: Anomaly scores
        manual_value: Manual threshold value (optional)
        
    Returns:
        Selected threshold value
    """
    precision, recall, thresholds = precision_recall_curve(y_true, errors)
    if len(thresholds) != len(precision) - 1:
        thresholds = thresholds[:len(precision) - 1]

    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    avg_scores = (precision[:-1] + recall[:-1]) / 2
    youden_scores = recall[:-1] - (1 - precision[:-1])

    threshold_dict = {}
    if manual_value is not None:
        threshold_dict['manual'] = manual_value

    if len(thresholds) > 0:
        threshold_dict['f1'] = thresholds[np.argmax(f1_scores)]
        threshold_dict['avg'] = thresholds[np.argmax(avg_scores)]
        threshold_dict['youden'] = thresholds[np.argmax(youden_scores)]

    # Evaluate all thresholds
    summary = []
    for key, val in threshold_dict.items():
        y_pred = (errors > val).astype(int)
        precision_v = precision_score(y_true, y_pred, zero_division=0)
        recall_v = recall_score(y_true, y_pred, zero_division=0)
        f1_v = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        TN, FP, FN, TP = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        TPR = TP / (TP + FN + 1e-10)
        FPR = FP / (FP + TN + 1e-10)
        summary.append((key, val, TPR, FPR, TPR - FPR, f1_v))

    df_summary = pd.DataFrame(summary, columns=["mode", "threshold", "TPR", "FPR", "TPR-FPR", "F1"])
    print("\nThreshold Comparison Table:")
    print(df_summary.sort_values("F1", ascending=False).to_string(index=False))

    best_mode, best_thresh, *_ = max(summary, key=lambda x: x[5])  # Select by F1 score
    print(f"\nBest threshold by F1: [{best_mode}] {best_thresh:.6f}")
    return best_thresh