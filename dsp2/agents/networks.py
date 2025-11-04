from __future__ import annotations
import torch
import torch.nn as nn

class DQNNet(nn.Module):
    def __init__(self, input_dim: int, m_elevators: int, hidden: int = 128):
        super().__init__()
        self.m = m_elevators
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, self.m * 4)
        )

    def forward(self, x):
        # x: (B, input_dim)
        q = self.backbone(x)
        B = q.shape[0]
        return q.view(B, self.m, 4)

