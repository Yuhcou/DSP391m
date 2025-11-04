# DSP2 — Multi‑Elevator EGCS + DDQN

This repo contains a simple multi‑elevator (EGCS) simulator and a DDQN agent.

Quick start:

1) Create a venv and install dependencies

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2) Train

```
python -m dsp2.train.train --config dsp2\configs\default.yaml
```

3) Evaluate

```
python -m dsp2.train.eval --config dsp2\configs\default.yaml
```

Notes
- Environment is Gym-like but has no external dependency on gym.
- State size = M*N + 2N + 2M, actions per elevator in {0,1,2,3}.
- Reward penalizes number of waiting and in-car passengers.
- TensorBoard logs in `dsp2/logs`.
# Environment
n_floors: 10
m_elevators: 2
capacity: 8
dt: 1.0
lambda: 0.05
t_max: 3600
seed: 42
w_wait: 1.0
w_incar: 0.2

# Agent
gamma: 0.99
lr: 0.0001
batch_size: 64
replay_capacity: 200000
target_update_steps: 10000
# tau: 1.0 for hard, <1.0 for soft
tau: 1.0
epsilon_start: 1.0
epsilon_end: 0.05
decay_steps: 200000
grad_clip: 5.0

# Training
training_steps: 100000
warmup_steps: 64
log_interval: 1000
ckpt_interval: 50000
logdir: dsp2/logs
import numpy as np
import torch
from dsp2.env.egcs_env import EGCSEnv
from dsp2.agents.ddqn_agent import DDQNAgent, AgentConfig
from dsp2.agents.replay import ReplayBuffer


def test_network_shapes_and_action():
    env = EGCSEnv(n_floors=4, m_elevators=2, capacity=3, seed=1)
    agent = DDQNAgent(env.state_size, env.N, env.M, config=AgentConfig(batch_size=4))
    s = env.reset()
    mask = env.action_mask()
    a = agent.select_action(s, mask, epsilon=0.0)
    assert a.shape == (env.M,)
    assert ((a >= 0) & (a <= 3)).all()


def test_replay_and_train_step():
    env = EGCSEnv(n_floors=4, m_elevators=2, capacity=3, seed=2)
    agent = DDQNAgent(env.state_size, env.N, env.M, config=AgentConfig(batch_size=4))
    buf = ReplayBuffer(100, env.state_size, env.M, seed=0)
    s = env.reset()
    for _ in range(20):
        a = agent.select_action(s, env.action_mask(), epsilon=1.0)
        s2, r, done, info = env.step(a)
        buf.add(s, a, r, s2, done)
        s = env.reset() if done else s2
    batch = buf.sample(4)
    def mask_fn(st):
        return env.action_mask()  # simple stand-in; env mask ignores state arg here
    loss, q_mean = agent.train_step(batch, mask_fn)
    assert isinstance(loss, float)
    assert isinstance(q_mean, float)
import numpy as np
from dsp2.env.egcs_env import EGCSEnv

def test_state_size_and_reset():
    env = EGCSEnv(n_floors=5, m_elevators=2, capacity=4, seed=123)
    s = env.reset()
    assert s.shape[0] == env.state_size
    # positions and directions initially zero
    assert np.allclose(s[:2], 0)


def test_mask_boundaries():
    env = EGCSEnv(n_floors=3, m_elevators=1, capacity=2, seed=123)
    env.reset()
    env.positions[0] = 2  # top
    mask = env.action_mask()
    assert mask.shape == (1,4)
    assert mask[0,1] == False  # up illegal
    env.positions[0] = 0  # bottom
    mask = env.action_mask()
    assert mask[0,2] == False  # down illegal


def test_step_shapes():
    env = EGCSEnv(n_floors=5, m_elevators=2, capacity=4, seed=0)
    s = env.reset()
    a = np.array([0, 1], dtype=np.int64)
    s2, r, done, info = env.step(a)
    assert s2.shape == s.shape
    assert isinstance(r, float)
    assert isinstance(done, bool)
    assert 'n_waiting' in info
from __future__ import annotations
import os
import argparse
import yaml
import numpy as np
import torch
from dsp2.env.egcs_env import EGCSEnv
from dsp2.agents.ddqn_agent import DDQNAgent, AgentConfig
from dsp2.agents.masks import legal_action_mask_from_state


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def run_eval(cfg_path: str, seeds=(0,1,2), episodes_per_seed: int = 3):
    cfg = load_config(cfg_path)
    env = EGCSEnv(n_floors=cfg.get('n_floors', 10), m_elevators=cfg.get('m_elevators', 2), capacity=cfg.get('capacity', 8),
                  dt=cfg.get('dt', 1.0), lambda_fn=lambda t: cfg.get('lambda', 0.05), t_max=cfg.get('t_max', 1200))
    agent = DDQNAgent(env.state_size, env.N, env.M)
    # optionally load checkpoint
    ckpt = cfg.get('eval_ckpt')
    if ckpt and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location='cpu')
        agent.policy.load_state_dict(state['policy'])
        agent.target.load_state_dict(state['policy'])

    returns = []
    for seed in seeds:
        np.random.seed(seed)
        env.rng = np.random.default_rng(seed)
        for ep in range(episodes_per_seed):
            s = env.reset()
            done = False
            ep_ret = 0.0
            while not done:
                mask = env.action_mask()
                a = agent.select_action(s, mask, epsilon=0.0)
                s, r, done, info = env.step(a)
                ep_ret += r
            returns.append(ep_ret)
    print(f"Eval over {len(returns)} episodes: mean={np.mean(returns):.2f} +/- {np.std(returns):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=os.path.join('dsp2', 'configs', 'default.yaml'))
    args = parser.parse_args()
    run_eval(args.config)
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
    # lambda_fn returns rate per time for current step; convert to Bernoulli prob
    lam = float(lambda_fn(t_step))
    p = 1.0 - np.exp(-lam * dt)
    passengers: List[Passenger] = []
    # For simplicity, at most one passenger per floor per step with prob p
    floors = np.arange(n_floors)
    arrivals = rng.random(n_floors) < p
    for s in floors[arrivals]:
        # choose destination uniformly among other floors
        choices = np.delete(floors, s)
        dst = int(rng.choice(choices))
        direction = 1 if dst > s else -1
        passengers.append(Passenger(src=int(s), dst=dst, direction=direction, t_arrival=t_step))
    return passengers

