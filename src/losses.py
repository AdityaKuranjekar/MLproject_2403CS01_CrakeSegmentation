"""Losses.

Weighted binary cross-entropy (the proposed change), with p = sigmoid(logit),
y in {0,1}, N pixels and crack weight w >= 1:

    L_wCE = -(1/N) * sum_i [ w * y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ]

w = 1 gives ordinary cross-entropy (baseline / ablation). For one output
logit, binary CE is identical to 2-class softmax CE.

Soft Dice (add-on only, not part of the assignment's comparison):

    L_Dice = 1 - (2 * sum p_i y_i + eps) / (sum p_i + sum y_i + eps)
"""
import torch
import torch.nn.functional as F


def weighted_bce(logits, target, w: float = 1.0):
    pos_weight = torch.tensor([w], device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)


def soft_dice(logits, target, eps: float = 1.0):
    p = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    inter = (p * t).sum(1)
    return (1 - (2 * inter + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


def make_loss(name: str, w: float):
    if name == "wce":
        return lambda lo, t: weighted_bce(lo, t, w)
    if name == "wce_dice":
        return lambda lo, t: weighted_bce(lo, t, w) + soft_dice(lo, t)
    raise ValueError(name)
