"""4-conv CNN ported from CNN-2-2.ipynb (multi-class section).

Layout (no BN, no dropout — matches notebook):
  Conv3x3(1 -> 32) -> ReLU -> MaxPool2d(2)
  Conv3x3(32 -> 64) -> ReLU -> MaxPool2d(2)
  Conv3x3(64 -> 128) -> ReLU
  Conv3x3(128 -> 256) -> ReLU -> MaxPool2d(2)
  Flatten -> Linear(feat -> 64) -> ReLU -> Linear(64, num_classes)

Convs use 'valid' padding to match Keras' default in the notebook.
Default input is 1x68x136 (the dataset's cnn5 mode); the flattened
feature size is inferred via a dummy forward so the head adapts if
input dims change.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CnnDeep(nn.Module):
    def __init__(self, num_classes: int, in_h: int = 68, in_w: int = 136):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        with torch.no_grad():
            feat_dim = self.features(torch.zeros(1, 1, in_h, in_w)).flatten(1).shape[1]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))
