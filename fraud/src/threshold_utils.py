"""Threshold selection utilities — val-based, test-applied protocol.

Per audit: thresholds MUST be selected on validation set, then applied to test.
This prevents adaptive overfitting / circular gate problems.
"""
import numpy as np


def select_threshold_by_f2(y_true: np.ndarray, probs: np.ndarray,
                           target_fbeta: float = 2.0) -> float:
    """Select threshold that maximizes F-beta score on validation data.
    
    Args:
        y_true: Binary labels (validation set)
        probs: Predicted probabilities (validation set)
        target_fbeta: Beta for F-beta (2.0 = F2, emphasizes recall)
        
    Returns:
        Optimal threshold value
    """
    thresholds = np.linspace(0.5, 0.9999, 500)
    best_thresh, best_fbeta = 0.5, 0.0
    
    for thresh in thresholds:
        preds = probs >= thresh
        tp = np.sum(preds & (y_true == 1))
        fp = np.sum(preds & (y_true == 0))
        fn = np.sum((~preds) & (y_true == 1))
        
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        fbeta = (1 + target_fbeta**2) * prec * rec / max(
            target_fbeta**2 * prec + rec, 1e-12)
        
        if fbeta > best_fbeta:
            best_fbeta = fbeta
            best_thresh = thresh
            
    return best_thresh


def evaluate_at_threshold(y_true: np.ndarray, probs: np.ndarray,
                          threshold: float) -> dict:
    """Evaluate metrics at a fixed threshold (test-applied).
    
    Args:
        y_true: Binary labels (test set)
        probs: Predicted probabilities (test set)
        threshold: Pre-selected threshold (from val)
        
    Returns:
        Metrics dict with precision, recall, F2, FP, TP
    """
    preds = probs >= threshold
    tp = np.sum(preds & (y_true == 1))
    fp = np.sum(preds & (y_true == 0))
    fn = np.sum((~preds) & (y_true == 1))
    tn = np.sum((~preds) & (y_true == 0))
    
    prec = tp / max(tp + fp, 1e-12)
    rec = tp / max(tp + fn, 1e-12)
    fbeta = (1 + 2**2) * prec * rec / max(2**2 * prec + rec, 1e-12)
    
    return {
        'threshold': float(threshold),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
        'precision': float(prec),
        'recall': float(rec),
        'fbeta': float(fbeta),
        'lift_at_threshold': float((tp / max(tp + fp, 1)) / max(y_true.mean(), 1e-12))
    }
