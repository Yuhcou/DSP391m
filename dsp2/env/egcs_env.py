from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Any, Optional
from .sim_helpers import FloorQueues, Passenger, sample_arrivals
from .passenger_tracker import PassengerTracker
from .adaptive_rewards import AdaptiveRewardCalculator

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
                 penalty_normalize: bool = True,
                 # Adaptive reward parameters
                 use_adaptive_reward: bool = False,
                 baseline_config: Optional[str] = None,
                 baseline_weight: float = 0.5,
                 performance_bonus_scale: float = 1.0,
                 comparative_penalty_scale: float = 2.0,
                 curriculum_stage: int = 0,
                 use_dynamic_weights: bool = False):
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
        self.w_wait_base = w_wait  # Store original weights
        self.w_incar_base = w_incar
        # Reward shaping
        self.r_alight = float(r_alight)
        self.r_board = float(r_board)
        self.penalty_normalize = bool(penalty_normalize)
        self.norm_denom = max(1, self.N * self.capacity)
        self.rng = np.random.default_rng(seed)
        
        # Adaptive reward system
        self.use_adaptive_reward = use_adaptive_reward
        self.use_dynamic_weights = use_dynamic_weights
        self.passenger_tracker = PassengerTracker()
        
        if self.use_adaptive_reward and baseline_config:
            self.adaptive_calculator = AdaptiveRewardCalculator(
                config_name=baseline_config,
                baseline_weight=baseline_weight,
                performance_bonus_scale=performance_bonus_scale,
                comparative_penalty_scale=comparative_penalty_scale,
                curriculum_stage=curriculum_stage
            )
        else:
            self.adaptive_calculator = None
        
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
        
        # Reset passenger tracker
        self.passenger_tracker.reset()
        
        # Reset penalty weights to base values
        self.w_wait = self.w_wait_base
        self.w_incar = self.w_incar_base
        
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
        
        # Track passengers for AWT/AJT calculation
        self.passenger_tracker.register_arrivals(len(arrivals), self.t)
        if boarded_total > 0:
            self.passenger_tracker.register_boarding(boarded_total, self.t)
        if alighted_total > 0:
            self.passenger_tracker.register_alighting(alighted_total, self.t)
        
        # 5) Reward calculation
        # Key insight: The variance comes from the QUADRATIC relationship between
        # arrivals and accumulated penalties. Solution: Use smoother penalty function
        n_waiting = sum(len(q) for q in self.queues.up) + sum(len(q) for q in self.queues.down)
        n_incar = sum(len(ob) for ob in self.onboard)
        
        # Get current performance metrics
        awt = self.passenger_tracker.get_current_awt()
        ajt = self.passenger_tracker.get_current_ajt()
        
        # Apply dynamic weights if enabled
        if self.use_dynamic_weights and self.adaptive_calculator and awt > 0:
            self.w_wait, self.w_incar = self.adaptive_calculator.get_dynamic_penalty_weights(awt)
        
        # Calculate base reward with CAPPED penalties to balance learning
        # Key insight: LOWER cap + HIGHER weight = agent learns to avoid queues EARLY
        # If cap is 50, agent can lazily let queues grow to 40-50
        # Solution: Cap at 30 with stronger weight (0.40) to make 20-30 waiting HURT
        
        # Cap waiting penalty at 30 (not 50!) to force early action
        MAX_PENALTY_WAITING = 30
        MAX_PENALTY_INCAR = 15
        
        if self.penalty_normalize:
            wait_capped = min(n_waiting, MAX_PENALTY_WAITING)
            incar_capped = min(n_incar, MAX_PENALTY_INCAR)
            wait_term = (self.w_wait * wait_capped) / self.norm_denom
            incar_term = (self.w_incar * incar_capped) / self.norm_denom
        else:
            # Piecewise linear: full penalty up to cap, then constant
            wait_capped = min(n_waiting, MAX_PENALTY_WAITING)
            incar_capped = min(n_incar, MAX_PENALTY_INCAR)
            wait_term = self.w_wait * wait_capped
            incar_term = self.w_incar * incar_capped
        
        base_reward = - (wait_term + incar_term) + self.r_alight * alighted_total + self.r_board * boarded_total
        
        # Apply adaptive reward if enabled
        if self.use_adaptive_reward and self.adaptive_calculator and awt > 0:
            stats = self.passenger_tracker.get_statistics()
            service_rate = stats['total_served'] / max(1, stats['total_arrived'])
            reward = self.adaptive_calculator.calculate_total_adaptive_reward(
                base_reward, awt, ajt, n_waiting, n_incar, service_rate
            )
        else:
            reward = base_reward
        
        # 6) Time and done
        self.t += 1
        done = self.t >= self.T_max
        obs = self._build_state()
        
        # Build info dict
        info = {
            "n_waiting": n_waiting,
            "n_incar": n_incar,
            "arrivals": len(arrivals),
            "alighted": int(alighted_total),
            "boarded": int(boarded_total),
            "awt": awt,
            "ajt": ajt,
            "base_reward": float(base_reward),
            "adaptive_reward": float(reward - base_reward) if self.use_adaptive_reward else 0.0,
        }
        
        # Add adaptive reward info if enabled
        if self.use_adaptive_reward and self.adaptive_calculator and awt > 0:
            adaptive_info = self.adaptive_calculator.get_info_dict(awt, ajt)
            info.update(adaptive_info)
        
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
