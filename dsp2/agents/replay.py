from __future__ import annotations
from typing import Tuple
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity: int, state_shape: int, m_elevators: int, seed: int = 0):
        self.capacity = capacity
        self.state_shape = state_shape
        self.m = m_elevators
        self.rng = np.random.default_rng(seed)
        self.size = 0
        self.ptr = 0
        self.s = np.zeros((capacity, state_shape), dtype=np.float32)
        self.a = np.zeros((capacity, m_elevators), dtype=np.int64)
        self.r = np.zeros((capacity,), dtype=np.float32)
        self.s2 = np.zeros((capacity, state_shape), dtype=np.float32)
        self.d = np.zeros((capacity,), dtype=np.float32)

    def add(self, s: np.ndarray, a: np.ndarray, r: float, s2: np.ndarray, d: bool):
        i = self.ptr
        self.s[i] = s
        self.a[i] = a
        self.r[i] = r
        self.s2[i] = s2
        self.d[i] = float(d)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def can_sample(self, batch_size: int) -> bool:
        return self.size >= batch_size

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = self.rng.integers(0, self.size, size=batch_size)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]


