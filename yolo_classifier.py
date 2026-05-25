"""YOLO11n-backbone classifier.

First 10 layers of `yolo11n.pt` (frozen during training) feed an
AvgPool -> Linear(256, 128) -> ReLU -> Dropout(0.4) -> Linear(128, num_outputs)
head. Trained externally; the saved state_dict uses `head.4` as the final
Linear with shape (num_outputs, 128).

`num_outputs` is the literal head width:
  - num_outputs == 1: BCE-style binary head — forward() expands to 2 logits
    via `cat([-x, x])` so the project's argmax-based eval works unchanged
    (logit > 0 -> class 1).
  - num_outputs >= 2: plain CrossEntropy-style multi-class head.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from ultralytics import YOLO


class YOLOClassifier(nn.Module):
    def __init__(self, num_outputs: int):
        super().__init__()
        yolo_backbone = YOLO("yolo11n.pt").model
        self.features = nn.Sequential(*list(yolo_backbone.children())[0][:10])
        for p in self.features.parameters():
            p.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_outputs),
        )
        self.num_outputs = num_outputs

    def forward(self, x):
        out = self.head(self.pool(self.features(x)))
        if self.num_outputs == 1:
            return torch.cat([-out, out], dim=1)
        return out
