"""U-Net architecture."""

import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, i, o, d=0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1, bias=False),
            nn.BatchNorm2d(o),
            nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1, bias=False),
            nn.BatchNorm2d(o),
            nn.ReLU(inplace=True),
        )
        self.drop = nn.Dropout2d(d) if d else nn.Identity()

    def forward(self, x):
        return self.drop(self.layers(x))


class UNet2D(nn.Module):
    def __init__(self, in_channels=1, num_classes=6, base_channels=32, dropout=0.1):
        super().__init__()
        b = base_channels
        self.e1, self.p1 = DoubleConv(in_channels, b), nn.MaxPool2d(2)
        self.e2, self.p2 = DoubleConv(b, b * 2, dropout), nn.MaxPool2d(2)
        self.e3, self.p3 = DoubleConv(b * 2, b * 4, dropout), nn.MaxPool2d(2)
        self.b = DoubleConv(b * 4, b * 8, dropout)
        self.u3, self.d3 = nn.ConvTranspose2d(b * 8, b * 4, 2, 2), DoubleConv(
            b * 8, b * 4, dropout
        )
        self.u2, self.d2 = nn.ConvTranspose2d(b * 4, b * 2, 2, 2), DoubleConv(
            b * 4, b * 2, dropout
        )
        self.u1, self.d1 = nn.ConvTranspose2d(b * 2, b, 2, 2), DoubleConv(b * 2, b)
        self.out = nn.Conv2d(b, num_classes, 1)

    def _up(self, x, ref):
        return (
            F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
            if x.shape[-2:] != ref.shape[-2:]
            else x
        )

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.p1(e1))
        e3 = self.e3(self.p2(e2))
        b = self.b(self.p3(e3))
        d3 = self.d3(torch.cat((self._up(self.u3(b), e3), e3), 1))
        d2 = self.d2(torch.cat((self._up(self.u2(d3), e2), e2), 1))
        return self.out(self.d1(torch.cat((self._up(self.u1(d2), e1), e1), 1)))
