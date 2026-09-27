from __future__ import annotations

import numpy as np
from sklearn import metrics


def tpr_at_x_fpr(no_w_scores, w_score, x=0.01):
    """Computes the TPR@x%FPR given scores for the watermark.

    Args:
        no_w_scores: Scores for the clean images
        w_score: Scores for the watermarked images
        x: FPR threshold. Default: 0.01.

    Returns:
        AUC, Accuracy, TPR@x%FPR
    """
    preds = no_w_scores + w_score
    t_labels = [0] * len(no_w_scores) + [1] * len(w_score)

    fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)
    low = tpr[np.where(fpr <= x)[0][-1]]

    return auc, acc, low


