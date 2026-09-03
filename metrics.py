"""Validation metrics."""

import numpy as np
import torch


@torch.no_grad()
def batch_metrics(logits, targets, num_classes):
    pred = logits.argmax(1)
    dices = []
    ious = []
    for c in range(num_classes):
        a, b = pred == c, targets == c
        inter = (a & b).sum().float()
        dices.append(float((2 * inter + 1e-5) / (a.sum() + b.sum() + 1e-5)))
        ious.append(float((inter + 1e-5) / ((a | b).sum() + 1e-5)))
    return dices, ious


def aggregate(items):
    dice = np.mean([x[0] for x in items], axis=0)
    iou = np.mean([x[1] for x in items], axis=0)
    return {
        "dice_per_class": dice.tolist(),
        "iou_per_class": iou.tolist(),
        "mean_dice_fg": float(dice[1:].mean()) if len(dice) > 1 else float(dice.mean()),
        "mean_iou_fg": float(iou[1:].mean()) if len(iou) > 1 else float(iou.mean()),
    }
