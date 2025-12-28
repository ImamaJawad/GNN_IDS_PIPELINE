"""
Evaluation metrics and scoring functions
Comprehensive model performance assessment
"""
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, classification_report)

def evaluate_model_comprehensive(y_true, y_pred, y_scores, model_name):
    """
    Comprehensive model evaluation with all metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels  
        y_scores: Prediction scores
        model_name: Name for display
        
    Returns:
        Dictionary of all computed metrics
    """
    print(f"\nComprehensive Evaluation - {model_name}")
    print("="*70)
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        auc_roc = roc_auc_score(y_true, y_scores)
    except:
        auc_roc = 0.0
    
    # Confusion matrix metrics
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    TPR = TP / (TP + FN + 1e-10)
    FPR = FP / (FP + TN + 1e-10)
    TNR = TN / (TN + FP + 1e-10)  # Specificity
    FNR = FN / (FN + TP + 1e-10)
    
    # Additional metrics
    balanced_accuracy = (TPR + TNR) / 2
    mcc = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-10)
    
    print(f"Performance Metrics:")
    print(f"   Accuracy:         {accuracy:.4f}")
    print(f"   Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"   Precision:        {precision:.4f}")
    print(f"   Recall (TPR):     {recall:.4f}")
    print(f"   FPR (FPR):     {FPR:.4f}")
    print(f"   Specificity (TNR): {TNR:.4f}")
    print(f"   F1-Score:         {f1:.4f}")
    print(f"   AUC-ROC:          {auc_roc:.4f}")
    print(f"   MCC:              {mcc:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Benign', 'Malicious']))
    
    return {
        'accuracy': accuracy, 'balanced_accuracy': balanced_accuracy,
        'precision': precision, 'recall': recall, 'specificity': TNR,
        'f1': f1, 'auc_roc': auc_roc, 'tpr': TPR, 'fpr': FPR, 'fnr': FNR,
        'mcc': mcc, 'tpr_minus_fpr': TPR-FPR, 'confusion_matrix': cm
    }

def propagate_node_scores_to_flows(df, node_errors, ip_to_idx):
    """
    Propagate node-level anomaly scores to flow-level for evaluation
    
    Args:
        df: Network flow dataframe
        node_errors: Node-level anomaly scores
        ip_to_idx: IP to node index mapping
        
    Returns:
        Tuple of (flow_scores, flow_labels)
    """
    print("Propagating node scores to flows...")
    
    ip_to_score = {}
    for ip, idx in ip_to_idx.items():
        if idx < len(node_errors):
            ip_to_score[ip] = node_errors[idx]
        else:
            ip_to_score[ip] = 0.0
    
    src_scores = df['IPV4_SRC_ADDR'].map(ip_to_score).fillna(0)
    dst_scores = df['IPV4_DST_ADDR'].map(ip_to_score).fillna(0)
    flow_scores = (src_scores + dst_scores) / 2
    
    flow_labels = df['Label'].values
    
    print(f"   Flow-level evaluation: {len(flow_scores)} flows")
    print(f"   Malicious flows: {np.sum(flow_labels)} ({np.mean(flow_labels)*100:.1f}%)")
    
    return flow_scores.values, flow_labels