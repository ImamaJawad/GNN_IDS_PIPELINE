"""
Visualization functions for model evaluation and results analysis
Creates comprehensive plots for performance assessment
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
from scipy.stats import mannwhitneyu

def create_comprehensive_plots(flow_labels, flow_scores, threshold, metrics, name, training_losses=None):
    """
    Create comprehensive visualization suite for model evaluation
    
    Args:
        flow_labels: True labels for flows
        flow_scores: Predicted anomaly scores
        threshold: Selected threshold
        metrics: Dictionary of computed metrics
        name: Dataset/model name for plots
        training_losses: Training loss history (optional)
    """
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots using a simpler approach
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(f'Comprehensive Analysis: {name}', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # 1. Precision-Recall Curve
    _plot_precision_recall_curve(axes[0], flow_labels, flow_scores, name)
    
    # 2. ROC Curve  
    _plot_roc_curve(axes[1], flow_labels, flow_scores)
    
    # 3. Error Distribution
    _plot_error_distribution(axes[2], flow_labels, flow_scores, threshold)
    
    # 4. Performance Metrics Bar Chart
    _plot_performance_metrics(axes[3], metrics)
    
    # 5. Confusion Matrix Heatmap
    _plot_confusion_matrix(axes[4], metrics)
    
    # 6. Training Loss Curve
    _plot_training_loss(axes[5], training_losses)
    
    # 7. Score vs Label Scatter
    _plot_score_scatter(axes[6], flow_labels, flow_scores, threshold)
    
    # 8. Class Statistics
    _plot_class_statistics(axes[7], flow_labels, flow_scores)
    
    # Hide unused subplot
    axes[8].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def _plot_precision_recall_curve(ax, flow_labels, flow_scores, name):
    """Plot precision-recall curve"""
    precision, recall, _ = precision_recall_curve(flow_labels, flow_scores)
    auc_pr = np.trapz(precision, recall)
    ax.plot(recall, precision, marker='.', linewidth=2, color='blue', label=f'AUC-PR: {auc_pr:.3f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve\n{name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

def _plot_roc_curve(ax, flow_labels, flow_scores):
    """Plot ROC curve"""
    fpr_roc, tpr_roc, _ = roc_curve(flow_labels, flow_scores)
    roc_auc = roc_auc_score(flow_labels, flow_scores)
    ax.plot(fpr_roc, tpr_roc, marker='.', linewidth=2, color='red', label=f'ROC AUC: {roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

def _plot_error_distribution(ax, flow_labels, flow_scores, threshold):
    """Plot error distribution by class"""
    benign_scores = flow_scores[flow_labels == 0]
    malicious_scores = flow_scores[flow_labels == 1]
    ax.hist(benign_scores, bins=50, alpha=0.7, label=f'Benign (n={len(benign_scores)})', color='green', density=True)
    ax.hist(malicious_scores, bins=50, alpha=0.7, label=f'Malicious (n={len(malicious_scores)})', color='red', density=True)
    ax.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.4f}')
    ax.set_xlabel('Reconstruction Error')
    ax.set_ylabel('Density')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

def _plot_performance_metrics(ax, metrics):
    """Plot performance metrics bar chart"""
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1', 'AUC-ROC']
    metrics_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], 
                     metrics['specificity'], metrics['f1'], metrics['auc_roc']]
    colors = plt.cm.Set3(np.linspace(0, 1, len(metrics_names)))
    
    bars = ax.bar(metrics_names, metrics_values, color=colors)
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics')
    ax.set_ylim(0, 1)
    
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.setp(ax.get_xticklabels(), rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

def _plot_confusion_matrix(ax, metrics):
    """Plot confusion matrix heatmap"""
    cm = metrics['confusion_matrix']
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
    ax.set_title('Normalized Confusion Matrix')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Set tick labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Benign', 'Malicious'])
    ax.set_yticklabels(['Benign', 'Malicious'])
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, f'{cm_normalized[i, j]:.3f}',
                         ha="center", va="center", color="black", fontweight='bold')

def _plot_training_loss(ax, training_losses):
    """Plot training loss curve"""
    if training_losses is not None and len(training_losses) > 0:
        epochs = range(1, len(training_losses) + 1)
        ax.plot(epochs, training_losses, 'b-', linewidth=2, label='Training Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Training Loss\nNot Available', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12)
        ax.set_title('Training Loss Curve')
        ax.set_xticks([])
        ax.set_yticks([])

def _plot_score_scatter(ax, flow_labels, flow_scores, threshold):
    """Plot scatter of scores vs labels"""
    benign_idx = flow_labels == 0
    malicious_idx = flow_labels == 1
    
    ax.scatter(range(np.sum(benign_idx)), flow_scores[benign_idx], 
              alpha=0.6, c='green', label='Benign', s=20)
    ax.scatter(range(np.sum(benign_idx), len(flow_scores)), flow_scores[malicious_idx], 
              alpha=0.6, c='red', label='Malicious', s=20)
    
    ax.axhline(threshold, color='black', linestyle='--', linewidth=2, 
               label=f'Threshold: {threshold:.4f}')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Anomaly Score')
    ax.set_title('Anomaly Scores by Class')
    ax.legend()
    ax.grid(True, alpha=0.3)

def _plot_class_statistics(ax, flow_labels, flow_scores):
    """Plot class statistics"""
    benign_scores = flow_scores[flow_labels == 0]
    malicious_scores = flow_scores[flow_labels == 1]
    
    stats_data = [
        ['Benign', len(benign_scores), np.mean(benign_scores), np.std(benign_scores), np.median(benign_scores)],
        ['Malicious', len(malicious_scores), np.mean(malicious_scores), np.std(malicious_scores), np.median(malicious_scores)]
    ]
    
    # Create table
    table = ax.table(cellText=[[f'{val:.4f}' if isinstance(val, float) else str(val) for val in row] 
                              for row in stats_data],
                    colLabels=['Class', 'Count', 'Mean', 'Std', 'Median'],
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    ax.axis('off')
    ax.set_title('Class Statistics')

# Alternative function for simpler visualization if the above still has issues
def create_simple_plots(flow_labels, flow_scores, threshold, metrics, name, training_losses=None):
    """
    Create a simpler visualization suite that avoids GridSpec issues
    """
    # Create individual plots
    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig1.suptitle(f'Model Performance: {name}', fontsize=14, fontweight='bold')
    
    # Plot 1: ROC Curve
    _plot_roc_curve(ax1, flow_labels, flow_scores)
    
    # Plot 2: Precision-Recall Curve  
    _plot_precision_recall_curve(ax2, flow_labels, flow_scores, name)
    
    # Plot 3: Error Distribution
    _plot_error_distribution(ax3, flow_labels, flow_scores, threshold)
    
    # Plot 4: Performance Metrics
    _plot_performance_metrics(ax4, metrics)
    
    plt.tight_layout()
    plt.show()
    
    # Second figure for additional plots
    fig2, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2, figsize=(12, 10))
    fig2.suptitle(f'Additional Analysis: {name}', fontsize=14, fontweight='bold')
    
    # Plot 5: Confusion Matrix
    _plot_confusion_matrix(ax5, metrics)
    
    # Plot 6: Training Loss
    _plot_training_loss(ax6, training_losses)
    
    # Plot 7: Score Scatter
    _plot_score_scatter(ax7, flow_labels, flow_scores, threshold)
    
    # Plot 8: Class Statistics
    _plot_class_statistics(ax8, flow_labels, flow_scores)
    
    plt.tight_layout()
    plt.show()
    
    return fig1, fig2