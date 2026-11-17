"""Pixel metrics for binary masks, pooled over every pixel of a split (micro).

TP = predicted crack & true crack, FP = predicted crack & true pavement,
FN = missed crack.  P = TP/(TP+FP), R = TP/(TP+FN), F1 = 2PR/(P+R),
IoU = TP/(TP+FP+FN).

`tolerant_counts` is an extra, commonly used for thin cracks (CrackForest,
DeepCrack papers): a predicted pixel counts as correct if a true crack pixel
lies within `tol` px, and vice versa. It is reported alongside the strict
scores, never instead of them.
"""
import numpy as np
from scipy.ndimage import binary_dilation


def counts(pred: np.ndarray, gt: np.ndarray):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    return int(tp), int(fp), int(fn)


def tolerant_counts(pred: np.ndarray, gt: np.ndarray, tol: int = 2):
    """pred, gt: (B,H,W). Returns (tp_precision, n_pred, tp_recall, n_gt)."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    st = np.zeros((1, 2 * tol + 1, 2 * tol + 1), bool)
    yy, xx = np.mgrid[-tol:tol + 1, -tol:tol + 1]
    st[0] = yy ** 2 + xx ** 2 <= tol ** 2
    gt_d, pred_d = binary_dilation(gt, st), binary_dilation(pred, st)
    return int((pred & gt_d).sum()), int(pred.sum()), int((gt & pred_d).sum()), int(gt.sum())


def scores(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {"precision": p, "recall": r, "f1": f1, "iou": iou}


def evaluate(pred: np.ndarray, gt: np.ndarray, tol: int = 2):
    out = scores(*counts(pred, gt))
    tpp, npred, tpr, ngt = tolerant_counts(pred, gt, tol)
    p = tpp / npred if npred else 0.0
    r = tpr / ngt if ngt else 0.0
    out[f"f1_tol{tol}"] = 2 * p * r / (p + r) if p + r else 0.0
    return out
