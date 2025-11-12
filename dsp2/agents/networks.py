from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional

class DQNNet(nn.Module):
    def __init__(self, input_dim: int, m_elevators: int, hidden: int = 256, dueling: bool = False):
        super().__init__()
        self.m = m_elevators
        self.dueling = dueling
        if not dueling:
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, self.m * 4)
            )
        else:
            # Shared feature extractor
            self.feature = nn.Sequential(
                nn.Linear(input_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
            )
            # Dueling streams: value per elevator (scalar) and advantage per action (4)
            self.value_head = nn.Linear(hidden, self.m)            # (B, M)
            self.adv_head = nn.Linear(hidden, self.m * 4)         # (B, M*4)

    def forward(self, x):
        # x: (B, input_dim)
        B = x.shape[0]
        if not getattr(self, 'dueling', False):
            q = self.backbone(x)
            return q.view(B, self.m, 4)
        # Dueling
        h = self.feature(x)
        v = self.value_head(h).view(B, self.m, 1)         # (B, M, 1)
        adv = self.adv_head(h).view(B, self.m, 4)         # (B, M, 4)
        adv_mean = adv.mean(dim=-1, keepdim=True)
        q = v + (adv - adv_mean)
        return q


class VDNMixer(nn.Module):
    """
    Simple mixing network to enable CTDE with VDN-style monotonic mixing.
    It outputs a scalar bias b(s) from the global state and adds it to the sum of per-agent Qs.
    If disabled, behaves as pure VDN (sum only).
    """
    def __init__(self, state_dim: int, use_central_bias: bool = False, hidden: int = 64):
        super().__init__()
        self.use_central_bias = use_central_bias
        if use_central_bias:
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, 1)
            )
        else:
            self.net = None

    def forward(self, q_per_elevator: torch.Tensor, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        # q_per_elevator: (B, M) values selected for each elevator
        # state: (B, S) global state features
        q_sum = q_per_elevator.sum(dim=-1, keepdim=True)  # (B,1)
        if self.use_central_bias:
            assert state is not None, "State tensor required when central bias is enabled"
            bias = self.net(state)  # (B,1)
            return q_sum + bias
        return q_sum
