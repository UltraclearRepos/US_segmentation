"""Loss functions for multiclass segmentation."""

import torch
from torch import nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, targets):
        probs = torch.softmax(logits, 1)
        one_hot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
        inter = (probs * one_hot).sum((0, 2, 3))
        total = (probs + one_hot).sum((0, 2, 3))
        dice = (2 * inter + 1e-5) / (total + 1e-5)
        return 1 - dice[1:].mean() if self.num_classes > 1 else 1 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, class_weights, num_classes, config):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss(num_classes)
        self.config = config

    def forward(self, logits, targets):
        ce = self.ce(logits, targets)
        dice = self.dice(logits, targets)
        raw = F.cross_entropy(logits, targets, weight=self.ce.weight, reduction="none")
        focal = ((1 - torch.exp(-raw)) ** self.config["focal_gamma"] * raw).mean()
        return (
            self.config["cross_entropy_weight"] * ce
            + self.config["dice_weight"] * dice
            + self.config["focal_weight"] * focal
        )
