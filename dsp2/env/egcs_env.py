from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Any, Optional
from .sim_helpers import FloorQueues, Passenger, sample_arrivals

class EGCSEnv:
    """
    A lightweight Gym-like multi-elevator environment.
    State vector order:
      positions (M,), directions (M,), hall_up (N,), hall_down (N,), car_calls (M*N,)
    Actions per elevator: 0=Stay, 1=Up, 2=Down, 3=Open
    """
    def __init__(self, n_floors: int = 10, m_elevators: int = 2, capacity: int = 8,
                 dt: float = 1.0, lambda_fn=lambda t: 0.05, t_max: int = 3600,
                 seed: Optional[int] = None,
                 w_wait: float = 1.0, w_incar: float = 0.2,
                 r_alight: float = 0.1, r_board: float = 0.02,
                 penalty_normalize: bool = True):
        assert n_floors >= 2
        assert m_elevators >= 1
        self.N = n_floors
        self.M = m_elevators
        self.capacity = capacity
        self.dt = dt
        self.lambda_fn = lambda_fn
        self.T_max = t_max
        self.w_wait = w_wait
        self.w_incar = w_incar
        # Reward shaping
        self.r_alight = float(r_alight)
        self.r_board = float(r_board)
        self.penalty_normalize = bool(penalty_normalize)
        self.norm_denom = max(1, self.N * self.capacity)
        self.rng = np.random.default_rng(seed)
        # dynamic state
        self.t = 0
        self.positions = np.zeros(self.M, dtype=np.int32)  # start at floor 0
        self.directions = np.zeros(self.M, dtype=np.int32)
        self.queues = FloorQueues(self.N)
        self.onboard = [list() for _ in range(self.M)]  # list of Passenger
        self.car_calls = np.zeros((self.M, self.N), dtype=np.float32)
        self._last_wait_sum = 0

    @property
    def state_size(self) -> int:
        return self.M * self.N + 2 * self.N + 2 * self.M

    def reset(self) -> np.ndarray:
        self.t = 0
        self.positions[:] = 0
        self.directions[:] = 0
        self.queues = FloorQueues(self.N)
        self.onboard = [list() for _ in range(self.M)]
        self.car_calls.fill(0)
        return self._build_state()

    def _build_state(self) -> np.ndarray:
        hall_up, hall_down = self.queues.hall_calls()
        pos = self.positions.astype(np.float32)
        dirs = self.directions.astype(np.float32)
        state = np.concatenate([
            pos, dirs, hall_up, hall_down, self.car_calls.flatten().astype(np.float32)
        ]).astype(np.float32)
        return state

    def action_mask(self) -> np.ndarray:
        # mask shape (M,4): True for legal actions
        mask = np.ones((self.M, 4), dtype=bool)
        top = self.positions >= (self.N - 1)
        bottom = self.positions <= 0
        if np.any(top):
            mask[top, 1] = False  # cannot go up
        if np.any(bottom):
            mask[bottom, 2] = False  # cannot go down
        # Open and Stay always legal
        return mask

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        assert action.shape == (self.M,), f"Action must be shape ({self.M},)"
        action = np.asarray(action, dtype=np.int64)
        # 1) Validate/mask illegal
        mask = self.action_mask()
        for i in range(self.M):
            if not mask[i, action[i]]:
                # clip to nearest legal: prefer Stay(0) then Open(3)
                action[i] = 0 if mask[i,0] else 3
        # 2) Apply movement effects
        alighted_total = 0
        boarded_total = 0
        for i in range(self.M):
            a = int(action[i])
            if a == 1 and self.positions[i] < self.N - 1:  # Up
                self.positions[i] += 1
                self.directions[i] = 1
            elif a == 2 and self.positions[i] > 0:  # Down
                self.positions[i] -= 1
                self.directions[i] = -1
            elif a == 0:  # Stay
                self.directions[i] = 0
            elif a == 3:  # Open doors
                # Direction becomes 0 when opening
                a_cnt, b_cnt = self._handle_open(i)
                alighted_total += a_cnt
                boarded_total += b_cnt
                self.directions[i] = 0
            else:
                # no-op if illegal
                self.directions[i] = 0
        # 4) Generate arrivals
        arrivals = sample_arrivals(self.N, self.dt, self.t, self.rng, self.lambda_fn)
        for p in arrivals:
            self.queues.add(p)
        # 5) Reward
        n_waiting = sum(len(q) for q in self.queues.up) + sum(len(q) for q in self.queues.down)
        n_incar = sum(len(ob) for ob in self.onboard)
        if self.penalty_normalize:
            wait_term = (self.w_wait * n_waiting) / self.norm_denom
            incar_term = (self.w_incar * n_incar) / self.norm_denom
        else:
            wait_term = self.w_wait * n_waiting
            incar_term = self.w_incar * n_incar
        reward = - (wait_term + incar_term) + self.r_alight * alighted_total + self.r_board * boarded_total
        # 6) Time and done
        self.t += 1
        done = self.t >= self.T_max
        obs = self._build_state()
        info = {
            "n_waiting": n_waiting,
            "n_incar": n_incar,
            "arrivals": len(arrivals),
            "alighted": int(alighted_total),
            "boarded": int(boarded_total),
        }
        return obs, float(reward), bool(done), info

    def _handle_open(self, i: int) -> Tuple[int, int]:
        floor = int(self.positions[i])
        # alight
        remaining = []
        alighted = 0
        for p in self.onboard[i]:
            if p.dst == floor:
                alighted += 1
                # passenger leaves, clear car call for this floor
            else:
                remaining.append(p)
        self.onboard[i] = remaining
        self.car_calls[i, floor] = 0.0
        # board
        free = self.capacity - len(self.onboard[i])
        boarded = 0
        if free > 0:
            # If elevator was moving last step, prefer that direction; otherwise None to board any
            last_dir = int(self.directions[i])
            dir_pref: Optional[int] = last_dir if last_dir != 0 else None
            boarded_list = self.queues.pop_for_boarding(floor, dir_pref, free)
            for p in boarded_list:
                self.onboard[i].append(p)
                self.car_calls[i, p.dst] = 1.0
                boarded += 1
        # if queues emptied for this floor, hall calls auto-clear via build_state
        return alighted, boarded
