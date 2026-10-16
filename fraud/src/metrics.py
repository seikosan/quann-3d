"""Fraud-detection metrics.

The dataset is extremely imbalanced (~0.12% fraud), so accuracy is meaningless.
Primary metric: AUC-PR (average precision). Also: AUC-ROC, and precision /
recall / F1 / F2 at an operational threshold. F2 weights false negatives (missed
fraud) 4x false positives — the usual cost asymmetry in fraud.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    fbeta_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def best_threshold(y: np.ndarray, prob: np.ndarray, beta: float = 2.0,
                   n_grid: int = 400) -> float:
    """Probability threshold that maximizes F-beta on `y`/`prob`.

    Grid search over quantile-distributed candidate thresholds (dense where the
    score moves). Robust to tiny / single-class sets. Returns the threshold.
    """
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob)
    if len(np.unique(y)) < 2:
        return 0.5
    qs = np.linspace(0.0, 1.0, n_grid)
    cands = np.unique(np.concatenate([np.quantile(prob, qs), [0.0, 1.0]]))
    pos = y == 1
    best_fb, best_thr = -1.0, 0.5
    for t in cands:
        pred = prob >= t
        tp = float(np.sum(pred & pos))
        fp = float(np.sum(pred & ~pos))
        fn = float(np.sum(~pred & pos))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = beta * beta * prec + rec
        fb = (1 + beta * beta) * prec * rec / denom if denom > 0 else 0.0
        if fb > best_fb:
            best_fb, best_thr = fb, float(t)
    return best_thr


def evaluate(y: np.ndarray, prob: np.ndarray, threshold: float | None = None,
             beta: float = 2.0) -> dict:
    """Full fraud metric set. If threshold is None, pick the best-F-beta one."""
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob)
    if threshold is None:
        threshold = best_threshold(y, prob, beta=beta)
    pred = (prob >= threshold).astype(int)

    out: dict = {
        "n": int(len(y)),
        "n_fraud": int(y.sum()),
        "base_rate": float(y.mean()),
        "threshold": float(threshold),
        "auc_pr": float(average_precision_score(y, prob)) if len(np.unique(y)) > 1 else float("nan"),
        "auc_roc": float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else float("nan"),
    }
    tn, fp, fn, tp = (confusion_matrix(y, pred, labels=[0, 1]).ravel()
                      if len(np.unique(y)) > 1 else (0, 0, 0, int(pred.sum())))
    out.update({
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(fbeta_score(y, pred, beta=1, zero_division=0)),
        "fbeta": float(fbeta_score(y, pred, beta=beta, zero_division=0)),
        "beta": beta,
        "predicted_fraud_rate": float(pred.mean()),
        "lift_at_threshold": _lift(y, pred),
    })
    return out


def _lift(y: np.ndarray, pred: np.ndarray) -> float:
    """Observed fraud rate among predicted-fraud / base rate."""
    base = y.mean()
    if base <= 0:
        return float("nan")
    hit = y[pred == 1].mean() if (pred == 1).any() else 0.0
    return float(hit / base)


def format_metrics(m: dict) -> str:
    b = m.get("beta", 2.0)
    blabel = f"F{b:g}"
    return (
        f"n={m['n']:,} (fraud {m['n_fraud']:,}, base {m['base_rate']:.4%})  "
        f"AUC-PR={m['auc_pr']:.4f}  AUC-ROC={m['auc_roc']:.4f}  "
        f"@thr={m['threshold']:.3f}: P={m['precision']:.3f} R={m['recall']:.3f} "
        f"F1={m['f1']:.3f} {blabel}={m['fbeta']:.3f}  lift={m['lift_at_threshold']:.1f}x  "
        f"(pred {m['predicted_fraud_rate']:.2%})"
    )
