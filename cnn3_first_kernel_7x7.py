"""3-layer CNN for 1x68x136 word images, with a 7x7 first conv.

Layout:
  block i: Conv(k_i)(c_i_in -> c_i_out) -> BN -> ReLU -> MaxPool2d(2)
  kernels:  7 -> 3 -> 3
  channels: 1 -> 32 -> 64 -> 128
  spatial:  68x136 -> 34x68 -> 17x34 -> 8x17
  head:     AdaptiveAvgPool2d(1) -> Dropout -> Linear(128, num_classes)
"""
from __future__ import annotations

import torch.nn as nn

CHANNELS = (32, 64, 128)
KERNELS = (7, 3, 3)


def _block(c_in: int, c_out: int, kernel_size: int = 3) -> nn.Sequential:
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=padding, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class Cnn3(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.4):
        super().__init__()
        in_ch = 1
        layers = []
        for out_ch, kernel_size in zip(CHANNELS, KERNELS):
            layers.append(_block(in_ch, out_ch, kernel_size))
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(CHANNELS[-1], num_classes),
        )

    def forward(self, x):
        return self.head(self.pool(self.features(x)))
