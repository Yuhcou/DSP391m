# DSP2 — Multi-Elevator EGCS with Deep Reinforcement Learning

A sophisticated Elevator Group Control System (EGCS) using Double Deep Q-Networks (DDQN) with adaptive rewards and curriculum learning.

## 🚀 Quick Start

```bash
# 1. Setup environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt

# 2. Train agent
python run_training.py

# 3. Evaluate against baselines
python evaluate_traditional.py --config simple

# 4. View results
tensorboard --logdir=dsp2/logs
```

## 📋 Overview

DSP2 combines:
- **Multi-elevator simulation** with realistic passenger dynamics
- **DDQN agent** with dueling architecture and VDN mixing
- **Adaptive reward system** comparing against traditional algorithms
- **Curriculum learning** through staged difficulty progression
- **Traditional baselines** (Collective, Nearest Car, Sectoring, Fixed Priority)

### Key Features

✅ **Advanced RL Techniques:**
- Double DQN with dueling architecture
- Optional Prioritized Experience Replay (PER)
- Value Decomposition Networks (VDN) for multi-agent coordination
- Action masking for invalid moves

✅ **Intelligent Reward Design:**
- Capped penalties to prevent training instability
- Positive shaping rewards for good actions
- Adaptive bonuses based on performance vs baselines
- Dynamic penalty weight adjustment

✅ **Comprehensive Evaluation:**
- 5 traditional algorithms for comparison
- Real-time metrics (AWT, AJT, service rate)
- TensorBoard visualization
- Statistical analysis tools

## 📚 Documentation

**Start Here:**
- [Quick Start Guide](docs/QUICKSTART.md) - Get running in 5 minutes
- [Architecture Overview](docs/ARCHITECTURE.md) - System design and components
- [API Reference](docs/API_REFERENCE.md) - Complete API documentation
- [Reward Model Guide](docs/REWARD_MODEL.md) - Reward engineering and tuning

## 🏗️ System Architecture

### Environment (`dsp2/env/`)
- **EGCSEnv**: Multi-elevator simulation with Gym-like interface
- **PassengerTracker**: Real-time AWT/AJT calculation
- **AdaptiveRewardCalculator**: Performance-based reward adaptation
- **FloorQueues**: Passenger queue management with FIFO boarding

### Agent (`dsp2/agents/`)
- **DDQNAgent**: Double DQN with multiple improvements
- **DQNNet**: Neural network with dueling architecture
- **ReplayBuffer**: Experience replay with optional prioritization
- **VDNMixer**: Value decomposition for multi-agent coordination
- **Traditional Algorithms**: Collective, Nearest Car, Sectoring, Fixed Priority

### State & Action Space

**State Vector** (size = M×N + 2×N + 2×M):
```
[positions(M), directions(M), hall_up(N), hall_down(N), car_calls(M×N)]
```

**Action Space** (per elevator):
- `0`: Stay/Idle
- `1`: Move Up
- `2`: Move Down  
- `3`: Open Doors

**Example:** 10 floors, 2 elevators → state size = 54, action space = 4²

## 🎯 Performance Metrics

### Primary Metrics
- **AWT** (Average Waiting Time): Time passengers wait in hall queues
- **AJT** (Average Journey Time): Time from boarding to destination
- **Episode Return**: Cumulative reward (higher is better)
- **Service Rate**: Percentage of passengers served (target: >99%)

### Baseline Comparisons

For simple config (5 floors, 2 elevators, λ=0.05):

| Algorithm | AWT | AJT | Return | Service Rate |
|-----------|-----|-----|--------|--------------|
| Random | 8.45 | 15.23 | -3450 | 99.2% |
| Collective | 5.23 | 12.67 | -2135 | 99.5% |
| Nearest Car | **3.12** | **10.45** | **-1568** | 99.7% |
| Sectoring | 3.67 | 11.23 | -1789 | 99.6% |
| **DDQN (ours)** | **~3.0** | **~10.0** | **~-1500** | **99.8%** |

*DDQN achieves performance competitive with or better than best traditional algorithms*

## 🔧 Configuration

### Available Configs

**`quick.yaml`** - Fast testing (2-3 minutes)
```yaml
n_floors: 3
m_elevators: 1
training_steps: 2000
```

**`simple.yaml`** - Development (10-15 minutes)
```yaml
n_floors: 5
m_elevators: 2
training_steps: 10000
```

**`default.yaml`** - Production (1-2 hours)
```yaml
n_floors: 10
m_elevators: 2
training_steps: 100000
```

**`adaptive.yaml`** - With adaptive rewards (30-45 minutes)
```yaml
n_floors: 5
m_elevators: 2
training_steps: 30000
use_adaptive_reward: true
baseline_config: "simple"
curriculum_stage: 0
```

### Key Parameters

**Environment:**
```yaml
n_floors: 10          # Building size
m_elevators: 2        # Number of elevators
capacity: 8           # Passengers per elevator
lambda: 0.05          # Arrival rate (Poisson process)
t_max: 3600          # Episode duration (steps)
```

**Rewards:**
```yaml
w_wait: 1.0          # Waiting penalty weight
w_incar: 0.2         # In-car penalty weight
r_alight: 0.1        # Alighting reward
r_board: 0.02        # Boarding reward
```

**Agent:**
```yaml
lr: 0.0001           # Learning rate
gamma: 0.99          # Discount factor
batch_size: 64       # Mini-batch size
epsilon_decay: 200000 # Exploration decay steps
dueling: true        # Use dueling DQN
use_vdn: true        # Use VDN mixing
```

## 📊 Training & Evaluation

### Basic Training

```bash
# Train with default config
python run_training.py

# Monitor with TensorBoard
tensorboard --logdir=dsp2/logs
```

### Evaluate Traditional Algorithms

```bash
# Generate baseline metrics
python evaluate_traditional.py --config simple --episodes 10

# Outputs performance table and saves baseline YAML
# Creates: dsp2/baselines/simple_baseline.yaml
```

### Train with Adaptive Rewards

```bash
# 1. Generate baselines first
python evaluate_traditional.py --config simple

# 2. Edit run_training.py to use adaptive.yaml
# 3. Train with adaptive rewards
python run_training.py
```

### Final Evaluation

```bash
# Compare trained agent vs random baseline
python final_evaluation.py
```

## 🧪 Example Usage

### Training Loop

```python
from dsp2.env.egcs_env import EGCSEnv
from dsp2.agents.ddqn_agent import DDQNAgent, AgentConfig
from dsp2.agents.replay import ReplayBuffer

# Setup
env = EGCSEnv(n_floors=10, m_elevators=2)
agent = DDQNAgent(env.state_size, env.N, env.M)
buffer = ReplayBuffer(100000, env.state_size, env.M)

# Training loop
state = env.reset()
for step in range(100000):
    action = agent.select_action(state, env.action_mask())
    next_state, reward, done, info = env.step(action)
    buffer.add(state, action, reward, next_state, done)
    
    if buffer.can_sample(64):
        batch = buffer.sample(64)
        loss, q_mean, _ = agent.train_step(batch, mask_fn)
    
    state = next_state if not done else env.reset()
```

### Using Traditional Algorithms

```python
from dsp2.agents.traditional_algorithms import TraditionalAlgorithmAdapter

adapter = TraditionalAlgorithmAdapter('nearest_car', n_floors=10, 
                                     m_elevators=2, capacity=8)
env = EGCSEnv(n_floors=10, m_elevators=2)

state = env.reset()
done = False
while not done:
    action = adapter.select_action(state, env)
    state, reward, done, info = env.step(action)
```

## 🎓 Advanced Features

### Adaptive Rewards

Compares agent performance against traditional algorithm baselines:

**5 Strategies:**
1. Dynamic Baseline Thresholding (performance tiers)
2. Comparative Reward Shaping (smooth comparison)
3. Multi-Metric Performance Index (AWT + AJT + service rate)
4. Algorithm-Specific Competitive Rewards (beat specific algorithms)
5. Staged Difficulty Training (curriculum learning)

**Enable in config:**
```yaml
use_adaptive_reward: true
baseline_config: "simple"
baseline_weight: 0.5
curriculum_stage: 0  # 0=random, 1=collective, 2=nearest, 3=sectoring
```

### Curriculum Learning

Progressive difficulty through staged targets:
- **Stage 0**: Beat Random (AWT < 8.5)
- **Stage 1**: Beat Collective (AWT < 5.2)
- **Stage 2**: Beat Nearest Car (AWT < 3.1)
- **Stage 3**: Beat Sectoring (AWT < 2.8)

### Multi-Agent Coordination

VDN (Value Decomposition Networks) for centralized training:
```yaml
use_vdn: true         # Enable VDN mixing
use_central_bias: true # Add central coordination signal
```

## 📈 Results Visualization

TensorBoard tracks:
- Training loss and Q-values
- Episode returns (total and per-step average)
- Waiting and in-car passenger counts
- AWT/AJT over time
- Performance tiers and curriculum progress
- Adaptive reward components

## 🔍 Troubleshooting

### Agent not learning?
- Check epsilon decay (should reach ~0.1 mid-training)
- Verify reward scale (penalties not too harsh)
- Try simpler config (quick.yaml)
- Enable dueling architecture

### Training unstable?
- Reduce learning rate: `lr: 0.00005`
- Enable gradient clipping: `grad_clip: 5.0`
- Increase batch size: `batch_size: 128`
- Clip rewards: `reward = np.clip(reward, -10, 10)`

### High variance?
- Increase replay buffer: `replay_capacity: 200000`
- Larger batches: `batch_size: 128`
- Reduce traffic: `lambda: 0.01`

See [QUICKSTART.md](docs/QUICKSTART.md#troubleshooting) for detailed solutions.

## 📦 Dependencies

- Python 3.8+
- PyTorch 1.10+
- NumPy
- PyYAML
- TensorBoard

```bash
pip install torch numpy pyyaml tensorboard
```

## 🗂️ Project Structure

```
DSP391m/
├── dsp2/
│   ├── env/              # Environment components
│   │   ├── egcs_env.py
│   │   ├── adaptive_rewards.py
│   │   ├── passenger_tracker.py
│   │   └── sim_helpers.py
│   ├── agents/           # RL agents
│   │   ├── ddqn_agent.py
│   │   ├── networks.py
│   │   ├── replay.py
│   │   ├── masks.py
│   │   └── traditional_algorithms.py
│   ├── configs/          # YAML configurations
│   ├── baselines/        # Baseline metrics
│   └── logs/             # Training logs
├── docs/                 # Documentation
│   ├── QUICKSTART.md
│   ├── ARCHITECTURE.md
│   ├── API_REFERENCE.md
│   └── REWARD_MODEL.md
├── run_training.py       # Main training script
├── evaluate_traditional.py
├── final_evaluation.py
├── requirements.txt
└── README.md
```

## 🎯 Use Cases

- **Research**: Benchmark for MARL algorithms
- **Education**: Learning RL and multi-agent systems
- **Industry**: Building management system optimization
- **Simulation**: Traffic pattern analysis and planning

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional traditional algorithms
- Advanced MARL techniques (QMIX, QTRAN, MAPPO)
- Real-world traffic patterns
- Energy efficiency metrics
- GUI visualization

## 📝 License

This project is for educational and research purposes.

## 🔗 References

- [DDQN Paper](https://arxiv.org/abs/1509.06461)
- [Dueling DQN](https://arxiv.org/abs/1511.06581)
- [VDN](https://arxiv.org/abs/1706.05296)
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)

## 📧 Contact

For questions or issues, please open a GitHub issue or refer to the documentation.

---

**Happy Training! 🚀**
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

