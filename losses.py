"""Loss functions for multiclass segmentation."""

import torch
from torch import nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes, include_background, smooth=1e-6):
        super().__init__()

        self.num_classes = num_classes
        self.include_background = include_background
        self.smooth = smooth

    def forward(self, logits, targets):
        # [B, C, H, W]
        probs = F.softmax(logits, dim=1)

        # [B, H, W] -> [B, C, H, W]
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        if not self.include_background:
            probs = probs[:, 1:]
            targets_one_hot = targets_one_hot[:, 1:]

        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        denominator = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)

        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, class_weights, num_classes, config):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss(num_classes=num_classes, include_background=False, smooth=config["smooth"])
        self._lambda = config["lambda"]

    def forward(self, logits, targets):
        ce = self.ce(logits, targets)
        dice = self.dice(logits, targets)

        return self._lambda * ce + (1 - self._lambda) * dice
