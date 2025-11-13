# API Reference

Complete API documentation for the DSP2 system.

## Table of Contents
- [Environment API](#environment-api)
- [Agent API](#agent-api)
- [Traditional Algorithms API](#traditional-algorithms-api)
- [Utilities API](#utilities-api)

---

## Environment API

### EGCSEnv

Main multi-elevator simulation environment.

```python
class EGCSEnv:
    def __init__(
        self,
        n_floors: int = 10,
        m_elevators: int = 2,
        capacity: int = 8,
        dt: float = 1.0,
        lambda_fn: Callable[[int], float] = lambda t: 0.05,
        t_max: int = 3600,
        seed: Optional[int] = None,
        w_wait: float = 1.0,
        w_incar: float = 0.2,
        r_alight: float = 0.1,
        r_board: float = 0.02,
        penalty_normalize: bool = True,
        use_adaptive_reward: bool = False,
        baseline_config: Optional[str] = None,
        baseline_weight: float = 0.5,
        performance_bonus_scale: float = 1.0,
        comparative_penalty_scale: float = 2.0,
        curriculum_stage: int = 0,
        use_dynamic_weights: bool = False
    )
```

**Parameters:**
- `n_floors` (int): Number of floors in the building
- `m_elevators` (int): Number of elevators
- `capacity` (int): Maximum passenger capacity per elevator
- `dt` (float): Time step duration in seconds
- `lambda_fn` (Callable): Function returning arrival rate at time t
- `t_max` (int): Maximum episode duration in time steps
- `seed` (int, optional): Random seed for reproducibility
- `w_wait` (float): Penalty weight for waiting passengers
- `w_incar` (float): Penalty weight for passengers in elevators
- `r_alight` (float): Reward for passengers alighting
- `r_board` (float): Reward for passengers boarding
- `penalty_normalize` (bool): Whether to normalize penalties by capacity
- `use_adaptive_reward` (bool): Enable adaptive reward system
- `baseline_config` (str, optional): Baseline configuration name
- `baseline_weight` (float): Weight for adaptive reward components
- `performance_bonus_scale` (float): Scale factor for performance bonuses
- `comparative_penalty_scale` (float): Scale factor for comparative penalties
- `curriculum_stage` (int): Current curriculum learning stage (0-3)
- `use_dynamic_weights` (bool): Enable dynamic penalty weight adjustment

**Properties:**
- `state_size` (int): Dimension of state vector = M*N + 2*N + 2*M
- `N` (int): Number of floors
- `M` (int): Number of elevators
- `t` (int): Current time step

#### reset()

Reset environment to initial state.

```python
def reset(self) -> np.ndarray:
    """
    Returns:
        state (np.ndarray): Initial state vector of shape (state_size,)
    """
```

**Returns:**
- State vector with all elevators at floor 0, no passengers

**Example:**
```python
env = EGCSEnv(n_floors=10, m_elevators=2)
state = env.reset()
# state.shape = (54,) for 10 floors, 2 elevators
```

#### step()

Execute one time step with given actions.

```python
def step(
    self, 
    action: np.ndarray
) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
    """
    Args:
        action (np.ndarray): Action for each elevator, shape (M,)
        
    Returns:
        next_state (np.ndarray): Next state vector
        reward (float): Reward signal
        done (bool): Whether episode is finished
        info (dict): Additional information
    """
```

**Args:**
- `action` (np.ndarray): Integer array of shape (M,) with values in [0, 3]
  - 0: Stay/Idle
  - 1: Move Up
  - 2: Move Down
  - 3: Open Doors

**Returns:**
- `next_state`: State vector after action execution
- `reward`: Scalar reward (typically negative due to penalties)
- `done`: True if t >= t_max
- `info`: Dictionary with keys:
  - `n_waiting` (int): Current passengers waiting
  - `n_incar` (int): Current passengers in elevators
  - `arrivals` (int): New passenger arrivals this step
  - `alighted` (int): Passengers who reached destination
  - `boarded` (int): Passengers who entered elevator
  - `awt` (float): Current average waiting time
  - `ajt` (float): Current average journey time
  - `base_reward` (float): Reward before adaptive adjustments
  - `adaptive_reward` (float): Adaptive reward bonus/penalty

**Example:**
```python
state = env.reset()
action = np.array([3, 1])  # Elevator 0 opens, Elevator 1 moves up
next_state, reward, done, info = env.step(action)
print(f"Waiting: {info['n_waiting']}, Reward: {reward:.2f}")
```

#### action_mask()

Get mask of legal actions for current state.

```python
def action_mask(self) -> np.ndarray:
    """
    Returns:
        mask (np.ndarray): Boolean mask of shape (M, 4)
    """
```

**Returns:**
- Boolean array where `mask[i, a]` is True if action `a` is legal for elevator `i`

**Example:**
```python
mask = env.action_mask()
# mask[0, 1] = False if elevator 0 is at top floor (can't go up)
# mask[1, 2] = False if elevator 1 is at bottom floor (can't go down)
```

---

### PassengerTracker

Real-time passenger metrics tracking.

```python
class PassengerTracker:
    def __init__(self)
```

#### register_arrivals()

Register new passenger arrivals.

```python
def register_arrivals(self, n_arrivals: int, current_time: int) -> None:
    """
    Args:
        n_arrivals: Number of new arrivals
        current_time: Current simulation time step
    """
```

#### register_boarding()

Register passengers boarding elevators (FIFO).

```python
def register_boarding(self, n_boarded: int, current_time: int) -> None:
    """
    Args:
        n_boarded: Number of passengers boarding
        current_time: Current simulation time step
    """
```

#### register_alighting()

Register passengers reaching destination (FIFO).

```python
def register_alighting(self, n_alighted: int, current_time: int) -> None:
    """
    Args:
        n_alighted: Number of passengers alighting
        current_time: Current simulation time step
    """
```

#### get_current_awt()

Get current average waiting time.

```python
def get_current_awt(self) -> float:
    """
    Returns:
        float: Average waiting time in time steps (0.0 if no data)
    """
```

#### get_current_ajt()

Get current average journey time.

```python
def get_current_ajt(self) -> float:
    """
    Returns:
        float: Average journey time in time steps (0.0 if no data)
    """
```

#### get_statistics()

Get comprehensive statistics dictionary.

```python
def get_statistics(self) -> Dict[str, Any]:
    """
    Returns:
        dict: Statistics including:
            - awt, ajt, system_time
            - total_arrived, total_boarded, total_served
            - currently_waiting, currently_in_car
            - samples_awt, samples_ajt
    """
```

---

### AdaptiveRewardCalculator

Calculates adaptive rewards based on baseline comparisons.

```python
class AdaptiveRewardCalculator:
    def __init__(
        self,
        config_name: str = 'simple',
        baseline_weight: float = 0.5,
        performance_bonus_scale: float = 1.0,
        comparative_penalty_scale: float = 2.0,
        curriculum_stage: int = 0
    )
```

**Parameters:**
- `config_name`: Baseline configuration to load ('simple', 'mini', 'default')
- `baseline_weight`: Weight for baseline comparison (0-1)
- `performance_bonus_scale`: Multiplier for bonuses
- `comparative_penalty_scale`: Multiplier for penalties
- `curriculum_stage`: Current stage (0=random, 1=collective, 2=nearest, 3=sectoring)

#### calculate_total_adaptive_reward()

Calculate total adaptive reward combining all strategies.

```python
def calculate_total_adaptive_reward(
    self,
    base_reward: float,
    awt: float,
    ajt: float,
    n_waiting: int,
    n_incar: int,
    service_rate: float = 1.0
) -> float:
    """
    Args:
        base_reward: Original environment reward
        awt: Current average waiting time
        ajt: Current average journey time
        n_waiting: Number of waiting passengers
        n_incar: Number of passengers in elevators
        service_rate: Fraction of passengers served (0-1)
        
    Returns:
        float: Total reward with adaptive bonuses/penalties (clipped to [-10, 10])
    """
```

#### get_performance_tier()

Determine performance tier based on AWT.

```python
def get_performance_tier(self, awt: float) -> str:
    """
    Args:
        awt: Average waiting time
        
    Returns:
        str: One of 'excellent', 'good', 'average', 'poor'
    """
```

#### get_dynamic_penalty_weights()

Get adjusted penalty weights based on performance.

```python
def get_dynamic_penalty_weights(self, awt: float) -> Tuple[float, float]:
    """
    Args:
        awt: Current average waiting time
        
    Returns:
        tuple: (w_wait, w_incar) adjusted weights
    """
```

---

## Agent API

### DDQNAgent

Double Deep Q-Network agent with multiple improvements.

```python
class DDQNAgent:
    def __init__(
        self,
        state_size: int,
        n_floors: int,
        m_elevators: int,
        device: str = None,
        config: AgentConfig = AgentConfig()
    )
```

**Parameters:**
- `state_size`: Dimension of state vector
- `n_floors`: Number of floors (for masking)
- `m_elevators`: Number of elevators
- `device`: Device for torch ('cuda' or 'cpu', auto-detected if None)
- `config`: Agent configuration (see AgentConfig)

#### select_action()

Select action using epsilon-greedy policy.

```python
def select_action(
    self,
    state: np.ndarray,
    mask: np.ndarray,
    epsilon: float = None
) -> np.ndarray:
    """
    Args:
        state: State vector of shape (state_size,)
        mask: Action mask of shape (M, 4)
        epsilon: Exploration rate (uses schedule if None)
        
    Returns:
        np.ndarray: Action array of shape (M,) with values in [0, 3]
    """
```

**Example:**
```python
state = env.reset()
mask = env.action_mask()
action = agent.select_action(state, mask, epsilon=0.1)
```

#### train_step()

Perform one training step on a batch.

```python
def train_step(
    self,
    batch: Tuple[np.ndarray, ...],
    mask_next_fn: Callable[[np.ndarray], np.ndarray],
    per_indices: Optional[np.ndarray] = None,
    per_weights: Optional[np.ndarray] = None
) -> Tuple[float, float, Optional[np.ndarray]]:
    """
    Args:
        batch: (states, actions, rewards, next_states, dones)
        mask_next_fn: Function to get action masks for next states
        per_indices: Indices for PER update (optional)
        per_weights: Importance sampling weights (optional)
        
    Returns:
        tuple: (loss, q_mean, td_errors_for_per)
            - loss: Training loss value
            - q_mean: Mean Q-value for monitoring
            - td_errors_for_per: TD errors for priority update (or None)
    """
```

**Example:**
```python
batch = replay_buffer.sample(64)
def mask_fn(state):
    return legal_action_mask_from_state(state, n_floors, m_elevators)
loss, q_mean, _ = agent.train_step(batch, mask_fn)
```

#### epsilon()

Get current exploration rate.

```python
def epsilon(self) -> float:
    """
    Returns:
        float: Current epsilon based on linear decay schedule
    """
```

---

### AgentConfig

Configuration dataclass for DDQNAgent.

```python
@dataclass
class AgentConfig:
    # Core DQN
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 64
    target_update_steps: int = 10000
    tau: float = 1.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    decay_steps: int = 200000
    grad_clip: float = 5.0
    
    # Improvements
    dueling: bool = True
    use_per: bool = False
    per_alpha: float = 0.6
    per_beta: float = 0.4
    per_beta_increment: float = 0.0
    
    # Multi-Agent
    use_vdn: bool = True
    use_central_bias: bool = False
```

---

### DQNNet

Neural network for Q-value estimation.

```python
class DQNNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        m_elevators: int,
        hidden: int = 256,
        dueling: bool = False
    )
```

**Parameters:**
- `input_dim`: State vector dimension
- `m_elevators`: Number of elevators
- `hidden`: Hidden layer size
- `dueling`: Use dueling architecture

#### forward()

Forward pass through network.

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: State tensor of shape (batch, input_dim)
        
    Returns:
        torch.Tensor: Q-values of shape (batch, M, 4)
    """
```

---

### ReplayBuffer

Experience replay buffer for off-policy learning.

```python
class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        state_shape: int,
        m_elevators: int,
        seed: int = 0
    )
```

**Parameters:**
- `capacity`: Maximum buffer size
- `state_shape`: Dimension of state vector
- `m_elevators`: Number of elevators
- `seed`: Random seed for sampling

#### add()

Add transition to buffer.

```python
def add(
    self,
    s: np.ndarray,
    a: np.ndarray,
    r: float,
    s2: np.ndarray,
    d: bool
) -> None:
    """
    Args:
        s: State
        a: Action
        r: Reward
        s2: Next state
        d: Done flag
    """
```

#### sample()

Sample random batch.

```python
def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
    """
    Args:
        batch_size: Number of samples
        
    Returns:
        tuple: (states, actions, rewards, next_states, dones)
    """
```

#### can_sample()

Check if buffer has enough samples.

```python
def can_sample(self, batch_size: int) -> bool:
    """
    Args:
        batch_size: Required samples
        
    Returns:
        bool: True if buffer.size >= batch_size
    """
```

---

## Traditional Algorithms API

### TraditionalAlgorithmAdapter

Unified interface for traditional EGCS algorithms.

```python
class TraditionalAlgorithmAdapter:
    def __init__(
        self,
        algorithm_type: str,
        n_floors: int,
        m_elevators: int,
        capacity: int
    )
```

**Parameters:**
- `algorithm_type`: One of 'collective', 'nearest_car', 'sectoring', 'fixed_priority', 'random'
- `n_floors`: Number of floors
- `m_elevators`: Number of elevators
- `capacity`: Elevator capacity

#### select_action()

Select actions for all elevators.

```python
def select_action(
    self,
    state: np.ndarray,
    env: EGCSEnv
) -> np.ndarray:
    """
    Args:
        state: State vector from environment
        env: Environment instance (for internal state access)
        
    Returns:
        np.ndarray: Action array of shape (M,)
    """
```

**Example:**
```python
adapter = TraditionalAlgorithmAdapter('nearest_car', n_floors=10, m_elevators=2, capacity=8)
env = EGCSEnv(n_floors=10, m_elevators=2)
state = env.reset()
action = adapter.select_action(state, env)
```

---

### Individual Algorithm Classes

All dispatchers share the same interface:

```python
def select_action(
    self,
    state: np.ndarray,
    positions: np.ndarray,
    directions: np.ndarray,
    hall_up: np.ndarray,
    hall_down: np.ndarray,
    car_calls: np.ndarray,
    onboard_counts: List[int]
) -> np.ndarray:
    """
    Returns:
        np.ndarray: Actions of shape (M,)
    """
```

**Available Dispatchers:**
- `CollectiveControlDispatcher`
- `NearestCarDispatcher`
- `SectoringDispatcher`
- `FixedPriorityDispatcher`

---

## Utilities API

### Action Masking

#### legal_action_mask_from_state()

Compute legal action mask from state vector.

```python
def legal_action_mask_from_state(
    state: np.ndarray,
    n_floors: int,
    m_elevators: int
) -> np.ndarray:
    """
    Args:
        state: State vector
        n_floors: Number of floors
        m_elevators: Number of elevators
        
    Returns:
        np.ndarray: Boolean mask of shape (M, 4)
    """
```

**Example:**
```python
from dsp2.agents.masks import legal_action_mask_from_state

state = env.reset()
mask = legal_action_mask_from_state(state, env.N, env.M)
# mask[i, 1] = False if elevator i at top floor
# mask[i, 2] = False if elevator i at bottom floor
```

#### torch_mask_illegal()

Apply mask to Q-values tensor.

```python
def torch_mask_illegal(
    q_values: torch.Tensor,
    mask: torch.Tensor,
    illegal_value: float = -1e9
) -> torch.Tensor:
    """
    Args:
        q_values: Q-value tensor of shape (B, M, 4)
        mask: Boolean mask of shape (B, M, 4)
        illegal_value: Value to set for illegal actions
        
    Returns:
        torch.Tensor: Masked Q-values
    """
```

---

### Simulation Helpers

#### Passenger

Data class for passenger representation.

```python
@dataclass
class Passenger:
    src: int          # Source floor
    dst: int          # Destination floor
    direction: int    # -1 for down, +1 for up
    t_arrival: int    # Arrival time step
```

#### FloorQueues

Manages hall queues for all floors.

```python
class FloorQueues:
    def __init__(self, n_floors: int)
```

**Methods:**

```python
def add(self, p: Passenger) -> None:
    """Add passenger to appropriate queue"""

def pop_for_boarding(
    self, 
    floor: int, 
    direction: Optional[int], 
    k: int
) -> List[Passenger]:
    """
    Remove up to k passengers from floor queues
    Args:
        floor: Floor number
        direction: Preferred direction (None for any)
        k: Maximum passengers to board
    Returns:
        List of boarded passengers
    """

def hall_calls(self) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        tuple: (up_calls, down_calls) binary vectors of shape (N,)
    """

def counts(self) -> Tuple[int, int]:
    """
    Returns:
        tuple: (up_count, down_count) total passengers waiting
    """
```

#### sample_arrivals()

Generate passenger arrivals using Poisson process.

```python
def sample_arrivals(
    n_floors: int,
    dt: float,
    t_step: int,
    rng: np.random.Generator,
    lambda_fn: Callable[[int], float]
) -> List[Passenger]:
    """
    Args:
        n_floors: Number of floors
        dt: Time step size
        t_step: Current time step
        rng: Random number generator
        lambda_fn: Function returning arrival rate
        
    Returns:
        List[Passenger]: New arrivals this step
    """
```

---

## Configuration Loading

### load_config()

Load configuration from YAML file.

```python
def load_config(path: str) -> dict:
    """
    Args:
        path: Path to YAML configuration file
        
    Returns:
        dict: Configuration dictionary
    """
```

**Example:**
```python
config = load_config('dsp2/configs/default.yaml')
env = EGCSEnv(**{k: v for k, v in config.items() if k in ['n_floors', 'm_elevators', ...]})
```

---

## Type Hints

Common type aliases used throughout the codebase:

```python
from typing import Tuple, Dict, List, Optional, Callable, Any
import numpy as np
import torch

# State and action types
State = np.ndarray          # Shape: (state_size,)
Action = np.ndarray         # Shape: (M,)
ActionMask = np.ndarray     # Shape: (M, 4)
Reward = float
Done = bool
Info = Dict[str, Any]

# Batch types
StateBatch = np.ndarray     # Shape: (B, state_size)
ActionBatch = np.ndarray    # Shape: (B, M)
RewardBatch = np.ndarray    # Shape: (B,)

# Transitions
Transition = Tuple[State, Action, Reward, State, Done]
Batch = Tuple[StateBatch, ActionBatch, RewardBatch, StateBatch, np.ndarray]
```

---

## Error Handling

### Common Exceptions

**ValueError:**
- Invalid action values (not in [0, 3])
- Mismatched array shapes
- Unknown algorithm type

**AssertionError:**
- Action shape mismatch: `action.shape != (M,)`
- State out of bounds
- Invalid configuration values

**Example Error Handling:**
```python
try:
    action = agent.select_action(state, mask)
    next_state, reward, done, info = env.step(action)
except ValueError as e:
    print(f"Invalid action or state: {e}")
except AssertionError as e:
    print(f"Assertion failed: {e}")
```

---

## Examples

### Basic Training Loop

```python
from dsp2.env.egcs_env import EGCSEnv
from dsp2.agents.ddqn_agent import DDQNAgent, AgentConfig
from dsp2.agents.replay import ReplayBuffer
from dsp2.agents.masks import legal_action_mask_from_state

# Setup
env = EGCSEnv(n_floors=10, m_elevators=2, capacity=8)
config = AgentConfig(batch_size=64, lr=1e-4)
agent = DDQNAgent(env.state_size, env.N, env.M, config=config)
buffer = ReplayBuffer(100000, env.state_size, env.M)

# Training
state = env.reset()
for step in range(100000):
    # Act
    mask = env.action_mask()
    action = agent.select_action(state, mask)
    next_state, reward, done, info = env.step(action)
    
    # Store
    buffer.add(state, action, reward, next_state, done)
    state = next_state if not done else env.reset()
    
    # Train
    if buffer.can_sample(64):
        batch = buffer.sample(64)
        mask_fn = lambda s: legal_action_mask_from_state(s, env.N, env.M)
        loss, q_mean, _ = agent.train_step(batch, mask_fn)
```

### Evaluating Traditional Algorithm

```python
from dsp2.env.egcs_env import EGCSEnv
from dsp2.agents.traditional_algorithms import TraditionalAlgorithmAdapter

env = EGCSEnv(n_floors=10, m_elevators=2)
adapter = TraditionalAlgorithmAdapter('nearest_car', env.N, env.M, env.capacity)

state = env.reset()
episode_return = 0.0
done = False

while not done:
    action = adapter.select_action(state, env)
    state, reward, done, info = env.step(action)
    episode_return += reward

print(f"Episode return: {episode_return:.2f}")
print(f"Final AWT: {info['awt']:.2f}")
```

### Using Adaptive Rewards

```python
env = EGCSEnv(
    n_floors=10,
    m_elevators=2,
    use_adaptive_reward=True,
    baseline_config='simple',
    baseline_weight=0.5,
    curriculum_stage=1  # Stage 1: Beat Collective
)

state = env.reset()
# Training loop with adaptive rewards
# Environment automatically applies bonuses/penalties
```
