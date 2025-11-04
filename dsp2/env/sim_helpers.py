from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple, Optional
import numpy as np

@dataclass
class Passenger:
    src: int
    dst: int
    direction: int  # -1 for down, +1 for up
    t_arrival: int  # discrete time step index

class FloorQueues:
    def __init__(self, n_floors: int):
        self.n = n_floors
        self.up: List[Deque[Passenger]] = [deque() for _ in range(n_floors)]
        self.down: List[Deque[Passenger]] = [deque() for _ in range(n_floors)]

    def add(self, p: Passenger):
        if p.direction > 0:
            self.up[p.src].append(p)
        else:
            self.down[p.src].append(p)

    def pop_for_boarding(self, floor: int, direction: Optional[int], k: int) -> List[Passenger]:
        boarded: List[Passenger] = []
        if direction is None or direction >= 0:
            while k > 0 and self.up[floor]:
                boarded.append(self.up[floor].popleft())
                k -= 1
        if k > 0 and (direction is None or direction <= 0):
            while k > 0 and self.down[floor]:
                boarded.append(self.down[floor].popleft())
                k -= 1
        return boarded

    def hall_calls(self) -> Tuple[np.ndarray, np.ndarray]:
        up_calls = np.array([1 if len(self.up[f]) > 0 else 0 for f in range(self.n)], dtype=np.float32)
        down_calls = np.array([1 if len(self.down[f]) > 0 else 0 for f in range(self.n)], dtype=np.float32)
        return up_calls, down_calls

    def counts(self) -> Tuple[int, int]:
        up_c = sum(len(q) for q in self.up)
        down_c = sum(len(q) for q in self.down)
        return up_c, down_c


def sample_arrivals(n_floors: int, dt: float, t_step: int, rng: np.random.Generator,
                     lambda_fn) -> List[Passenger]:
    lam = float(lambda_fn(t_step))
    p = 1.0 - np.exp(-lam * dt)
    passengers: List[Passenger] = []
    floors = np.arange(n_floors)
    arrivals = rng.random(n_floors) < p
    for s in floors[arrivals]:
        choices = np.delete(floors, s)
        dst = int(rng.choice(choices))
        direction = 1 if dst > s else -1
        passengers.append(Passenger(src=int(s), dst=dst, direction=direction, t_arrival=t_step))
    return passengers

