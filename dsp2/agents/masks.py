from __future__ import annotations
import torch
import numpy as np

def legal_action_mask_from_state(state: np.ndarray, n_floors: int, m_elevators: int) -> np.ndarray:
    # state layout: pos(M), dir(M), hall_up(N), hall_down(N), car(M*N)
    pos = state[:m_elevators]
    pos = np.rint(pos).astype(int)
    mask = np.ones((m_elevators, 4), dtype=bool)
    mask[pos >= (n_floors - 1), 1] = False
    mask[pos <= 0, 2] = False
    return mask

def torch_mask_illegal(q_values: torch.Tensor, mask: torch.Tensor, illegal_value: float = -1e9) -> torch.Tensor:
    # q_values: (B, M, 4), mask: (B, M, 4) boolean
    masked = q_values.clone()
    masked[~mask] = illegal_value
    return masked

