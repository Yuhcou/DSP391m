# DSP2 - Multi-Elevator EGCS with DDQN

## Architecture Overview

DSP2 (Dispatcher System Project 2) is a reinforcement learning-based Elevator Group Control System (EGCS) that uses Double Deep Q-Networks (DDQN) to optimize multi-elevator dispatching. The system includes both traditional baseline algorithms and advanced RL agents with adaptive reward mechanisms.

## System Components

### 1. Environment (`dsp2/env/`)

The environment simulates a multi-elevator building with passenger arrivals, boarding, and alighting.

#### **EGCSEnv** (`egcs_env.py`)
Main simulation environment following a Gym-like interface.

**Key Features:**
- Multi-elevator coordination
- Real-time passenger simulation with Poisson arrival process
- Action masking for invalid moves
- Adaptive reward calculation
- Performance tracking (AWT, AJT)

**State Space:**
- **Size**: `M*N + 2*N + 2*M` where M=elevators, N=floors
- **Components**:
  - Elevator positions: `(M,)` - current floor of each elevator
  - Elevator directions: `(M,)` - movement direction (-1, 0, +1)
  - Hall up calls: `(N,)` - binary indicator for up requests per floor
  - Hall down calls: `(N,)` - binary indicator for down requests per floor
  - Car calls: `(M*N,)` - flattened matrix of destination requests per elevator

**Action Space:**
- **Per Elevator**: 4 discrete actions
  - `0`: Stay (idle at current floor)
  - `1`: Move Up (to next floor)
  - `2`: Move Down (to previous floor)
  - `3`: Open Doors (board/alight passengers)
- **Total**: Joint action space of size `4^M`
- **Masking**: Illegal actions (up at top floor, down at bottom) are masked

**Reward Structure:**

*Base Reward:*
```python
reward = -(w_wait * n_waiting + w_incar * n_incar) 
         + r_alight * passengers_alighted 
         + r_board * passengers_boarded
```

*Default Weights:*
- `w_wait = 1.0` - penalty weight for waiting passengers
- `w_incar = 0.2` - penalty weight for passengers in elevator
- `r_alight = 0.1` - positive reward for delivering passengers
- `r_board = 0.02` - positive reward for picking up passengers

*Key Mechanisms:*
- **Penalty Capping**: Waiting penalty capped at 30 passengers, in-car at 15
  - Prevents catastrophic negative rewards during learning
  - Encourages early intervention before queues grow
- **Penalty Normalization**: Optional division by `N * capacity` for scale invariance
- **Adaptive Rewards**: Optional comparison against traditional algorithm baselines

**Parameters:**
- `n_floors`: Number of building floors (default: 10)
- `m_elevators`: Number of elevators (default: 2)
- `capacity`: Max passengers per elevator (default: 8)
- `dt`: Time step size in seconds (default: 1.0)
- `lambda_fn`: Passenger arrival rate function (default: λ=0.05)
- `t_max`: Maximum episode duration in steps (default: 3600)
- `seed`: Random seed for reproducibility

#### **PassengerTracker** (`passenger_tracker.py`)
Real-time tracking of individual passenger metrics during training.

**Tracks:**
- Waiting passengers with arrival times
- Boarded passengers with board times
- Completed journeys with full time history

**Metrics Calculated:**
- **AWT** (Average Waiting Time): Mean time passengers wait in hall queues
- **AJT** (Average Journey Time): Mean time from boarding to alighting
- **System Time**: Total time = AWT + AJT
- **Service Rate**: Percentage of passengers successfully served

**Usage:**
```python
tracker = PassengerTracker()
tracker.register_arrivals(n_arrivals, current_time)
tracker.register_boarding(n_boarded, current_time)
tracker.register_alighting(n_alighted, current_time)
awt = tracker.get_current_awt()
ajt = tracker.get_current_ajt()
```

#### **AdaptiveRewardCalculator** (`adaptive_rewards.py`)
Calculates adaptive rewards by comparing agent performance against traditional algorithm baselines.

**Five Reward Adaptation Strategies:**

1. **Dynamic Baseline Thresholding**
   - Graduated bonuses based on performance tiers
   - Tiers: Excellent (beat best), Good (above average), Average, Poor
   - Bonus range: +2.0 to -1.0

2. **Comparative Reward Shaping**
   - Compare current AWT/AJT vs baseline mean
   - Uses tanh to bound bonuses: `tanh(1 - awt_ratio)`
   - Penalizes worse performance, rewards better

3. **Multi-Metric Performance Index**
   - Combines AWT, AJT, and service rate
   - Sigmoid-like scoring bounded to [-1, 1]
   - Weighted combination: AWT (0.2), AJT (0.1), Service (0.05)

4. **Algorithm-Specific Competitive Rewards**
   - Bonus for beating multiple baseline algorithms
   - +3.0 for beating all, +2.0 for most, +1.0 for half
   - Encourages progressive improvement

5. **Staged Difficulty Training (Curriculum)**
   - Progress through algorithm targets: Random → Collective → Nearest Car → Sectoring
   - Bonus for meeting current stage target
   - Clipped penalty (-1.0 max) for missing target

**Dynamic Weight Adjustment:**
```python
if awt <= excellent_threshold:
    w_wait, w_incar = 0.5, 0.1  # Reduce penalties
elif awt <= average_threshold:
    w_wait, w_incar = 1.5, 0.3  # Increase penalties
else:
    w_wait, w_incar = 2.0, 0.5  # Strong guidance
```

**Critical Clipping:**
- Individual strategies clipped to [-2, 2] range
- Total adaptive bonus clipped to [-5, 5]
- Final reward clipped to [-10, 10]
- Prevents reward explosion during poor performance

**Configuration:**
```python
calculator = AdaptiveRewardCalculator(
    config_name='simple',           # Baseline to load
    baseline_weight=0.5,            # Weight for baseline comparisons
    performance_bonus_scale=1.0,    # Multiplier for bonuses
    comparative_penalty_scale=2.0,  # Multiplier for penalties
    curriculum_stage=0              # Current curriculum stage
)
```

#### **Simulation Helpers** (`sim_helpers.py`)

**Passenger Class:**
```python
@dataclass
class Passenger:
    src: int          # Source floor
    dst: int          # Destination floor
    direction: int    # -1 for down, +1 for up
    t_arrival: int    # Arrival time step
```

**FloorQueues:**
- Maintains separate up/down queues per floor
- FIFO boarding logic
- Provides binary hall call indicators

**sample_arrivals():**
- Poisson arrival process: `P(arrival) = 1 - exp(-λ * dt)`
- Random source/destination floor assignment
- Returns list of new Passenger objects

### 2. Agent (`dsp2/agents/`)

Deep reinforcement learning agent using DDQN with various improvements.

#### **DDQNAgent** (`ddqn_agent.py`)
Main agent implementing Double DQN with multiple enhancements.

**Architecture Components:**
- **Policy Network**: Current Q-network for action selection
- **Target Network**: Stabilizes learning via delayed updates
- **VDN Mixer**: Optional centralized mixing for MARL (multi-agent)
- **Experience Replay**: Stores transitions for off-policy learning

**Key Improvements:**

1. **Dueling Architecture** (enabled by default)
   - Separates state value V(s) and action advantages A(s,a)
   - Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
   - Better value estimation, faster learning

2. **Double DQN**
   - Policy network selects actions: a* = argmax Q_policy(s', a)
   - Target network evaluates: Q_target(s', a*)
   - Reduces overestimation bias

3. **VDN Mixing** (optional)
   - Centralizes individual elevator Q-values
   - Q_total = Σ Q_i + bias(global_state)
   - Enables centralized training, decentralized execution (CTDE)

4. **Prioritized Experience Replay (PER)** (optional)
   - Sample transitions proportional to TD error
   - Focuses learning on surprising transitions
   - Uses importance sampling weights

**Training Loop:**
```python
# 1. Select action with epsilon-greedy
a = agent.select_action(state, mask, epsilon)

# 2. Store transition
buffer.add(state, action, reward, next_state, done)

# 3. Sample batch and train
batch = buffer.sample(batch_size)
loss, q_mean, td_errors = agent.train_step(batch, mask_fn)

# 4. Update target network (hard or soft)
if step % target_update_steps == 0:
    target.load_state_dict(policy.state_dict())
```

**Hyperparameters:**
- `gamma = 0.99` - discount factor
- `lr = 1e-4` - learning rate
- `batch_size = 64` - mini-batch size
- `replay_capacity = 200000` - buffer size
- `target_update_steps = 10000` - hard update frequency
- `tau = 1.0` - hard update (or soft if < 1.0)
- `epsilon_start = 1.0`, `epsilon_end = 0.05` - exploration schedule
- `decay_steps = 200000` - epsilon decay period
- `grad_clip = 5.0` - gradient clipping threshold

**Action Selection:**
```python
def select_action(state, mask, epsilon):
    if random() < epsilon:
        # Random legal action
        return random_choice(legal_actions)
    else:
        # Greedy action
        Q = policy(state)  # (M, 4)
        Q[~mask] = -inf     # Mask illegal
        return argmax(Q, axis=-1)  # (M,)
```

**TD Error Calculation:**
```python
# Current Q-values
Q_current = policy(s)[a]  # Q(s,a)

# Double DQN target
with torch.no_grad():
    Q_next_policy = policy(s')
    Q_next_policy[~mask] = -inf
    a_star = argmax(Q_next_policy)      # Policy selects
    Q_target = target(s')[a_star]       # Target evaluates
    y = r + gamma * Q_target * (1-done)

# TD error and loss
td_error = Q_current - y
loss = mean((td_error)^2)
```

#### **DQNNet** (`networks.py`)
Neural network architecture for Q-value estimation.

**Standard Architecture:**
```
Input(state_size) 
  → Linear(hidden=256) → ReLU
  → Linear(hidden=256) → ReLU
  → Linear(M*4) 
  → Reshape(M, 4)
```

**Dueling Architecture:**
```
Input(state_size)
  → Linear(hidden=256) → ReLU
  → Linear(hidden=256) → ReLU
  ├─→ Value Head: Linear(M) → (M, 1)
  └─→ Advantage Head: Linear(M*4) → (M, 4)
  
Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
```

**VDN Mixer:**
```python
class VDNMixer(nn.Module):
    def forward(self, q_per_elevator, global_state):
        q_sum = sum(q_per_elevator)  # VDN aggregation
        if use_central_bias:
            bias = MLP(global_state)  # Central coordination
            return q_sum + bias
        return q_sum
```

#### **ReplayBuffer** (`replay.py`)
Stores and samples transitions for off-policy learning.

**Structure:**
```python
buffer = {
    's': (capacity, state_size),      # States
    'a': (capacity, M),               # Actions
    'r': (capacity,),                 # Rewards
    's2': (capacity, state_size),     # Next states
    'd': (capacity,),                 # Done flags
}
```

**Operations:**
- `add(s, a, r, s2, done)` - Store transition (circular buffer)
- `sample(batch_size)` - Uniform random sampling
- `can_sample(batch_size)` - Check if enough samples

**PER Extension** (optional):
- SumTree data structure for efficient prioritized sampling
- Update priorities based on TD error: `priority = |td_error|^α`

#### **Action Masking** (`masks.py`)
Ensures agents only select valid actions.

```python
def legal_action_mask_from_state(state, n_floors, m_elevators):
    positions = state[:m_elevators]
    mask = ones((m_elevators, 4))
    mask[positions >= n_floors-1, 1] = False  # No up at top
    mask[positions <= 0, 2] = False           # No down at bottom
    return mask
```

### 3. Traditional Algorithms (`dsp2/agents/traditional_algorithms.py`)

Classic EGCS algorithms for baseline comparison and adaptive reward calculation.

#### **CollectiveControlDispatcher**
Simple up/down control - each elevator services calls in current direction until exhausted.

**Logic:**
1. If car call at current floor → Open doors
2. If moving up → Continue up while calls exist above
3. When no calls in direction → Reverse
4. If idle → Answer nearest call

**Pros:** Simple, low computation
**Cons:** Poor under heavy/unbalanced traffic

#### **NearestCarDispatcher**
Assigns hall calls to elevator that can serve soonest.

**Scoring Function (Otis-style):**
```python
if elevator_idle:
    score = N + 1 - distance * 0.5
elif same_direction and ahead_of_call:
    score = 2 * (N + 1 - distance)
else:
    score = 1
```

**Logic:**
1. Calculate score for each elevator-call pair
2. Assign call to highest-scoring elevator
3. Each elevator serves assigned calls + car calls
4. Move toward nearest assigned target

**Pros:** Better than collective under varied traffic
**Cons:** Doesn't consider full system state

#### **SectoringDispatcher**
Divides building into zones; each elevator primarily serves its zone.

**Zone Assignment:**
```python
floors_per_zone = n_floors / m_elevators
zone[i] = [i * floors_per_zone, (i+1) * floors_per_zone)
```

**Logic:**
1. Priority for calls within assigned zone
2. Help other zones if own zone is empty
3. Return to zone center when idle

**Pros:** Reduces travel distance, predictable
**Cons:** Inefficient if traffic is uneven across zones

#### **FixedPriorityDispatcher**
Similar to sectoring but with strict priority rules.

**Logic:**
1. Elevator i always serves car calls
2. Elevator i only serves hall calls in priority zone
3. Never helps other zones for hall calls
4. Return to zone center when idle

**Pros:** Very predictable, simple coordination
**Cons:** Not adaptive, can have idle elevators

#### **TraditionalAlgorithmAdapter**
Wrapper class to use traditional algorithms with EGCSEnv interface.

```python
adapter = TraditionalAlgorithmAdapter('nearest_car', n_floors, m_elevators, capacity)
action = adapter.select_action(state, env)
```

### 4. Training & Evaluation

#### **Training Script** (`run_training.py`)
Main training loop with TensorBoard logging.

**Pipeline:**
```
1. Load config (YAML)
2. Create environment and agent
3. Initialize replay buffer
4. Training loop:
   - Collect experience with epsilon-greedy
   - Sample batch from replay
   - Compute TD error and update
   - Log metrics to TensorBoard
   - Periodic checkpoint saving
5. Final evaluation
```

**Logged Metrics:**
- Loss and Q-value mean (per training step)
- Episode return (total and per-step average)
- Number waiting/in-car (per episode)
- AWT, AJT (if adaptive rewards enabled)
- Performance tier (if adaptive rewards enabled)
- Curriculum progress (if staged training)

**Key Training Parameters:**
```yaml
training_steps: 100000      # Total environment steps
warmup_steps: 64            # Steps before training starts
log_interval: 1000          # Logging frequency
ckpt_interval: 50000        # Checkpoint saving frequency
```

#### **Traditional Baseline Evaluation** (`evaluate_traditional.py`)
Comprehensive evaluation of traditional algorithms to establish baselines.

**Algorithms Evaluated:**
- Random (lower bound)
- Collective Control
- Nearest Car
- Sectoring
- Fixed Priority

**Metrics Collected:**
- Episode Return (cumulative reward)
- Average Waiting Time (AWT)
- Average Journey Time (AJT)
- Average System Time (AWT + AJT)
- Average Queue Size
- Average In-Car Count
- Passengers Served / Generated
- Service Rate

**Output:**
1. Performance comparison table
2. Statistical summary (mean ± std)
3. Best algorithm identification
4. Baseline YAML file for adaptive rewards

**Usage:**
```bash
python evaluate_traditional.py --config simple --episodes 10 --algorithms collective nearest_car sectoring
```

**Baseline File Format:**
```yaml
config: simple
algorithms:
  collective:
    mean_awt: 5.23
    std_awt: 0.45
    mean_ajt: 12.67
    ...
  nearest_car:
    mean_awt: 3.12
    ...
aggregate:
  mean_awt: 4.5
  min_awt: 2.8
  max_awt: 7.2
  ...
```

#### **Final Evaluation** (`final_evaluation.py`)
Compares trained DDQN agent against random baseline.

**Evaluation Protocol:**
1. Load trained checkpoint
2. Run 10 episodes with greedy policy (ε=0)
3. Run 10 episodes with random policy
4. Compare performance metrics
5. Report improvement

**Output:**
- Mean return and std for both policies
- Mean waiting/in-car counts
- Absolute improvement
- Success/failure verdict

### 5. Configuration System

#### **YAML Configuration Files** (`dsp2/configs/`)

**simple.yaml** - Minimal config for testing
- 5 floors, 2 elevators, capacity 6
- λ = 0.01, t_max = 1200
- 10K training steps

**mini.yaml** - Faster training
- 4 floors, 2 elevators, capacity 4
- λ = 0.02, t_max = 600
- 5K training steps

**quick.yaml** - Ultra-fast debugging
- 3 floors, 1 elevator, capacity 3
- λ = 0.05, t_max = 300
- 2K training steps

**default.yaml** - Full-scale training
- 10 floors, 2 elevators, capacity 8
- λ = 0.05, t_max = 3600
- 100K training steps

**adaptive.yaml** - With adaptive rewards
- 5 floors, 2 elevators, capacity 6
- λ = 0.05, t_max = 1200
- Adaptive rewards enabled with simple baseline
- 30K training steps

#### **Configuration Structure**

```yaml
# Environment
n_floors: 10
m_elevators: 2
capacity: 8
dt: 1.0
lambda: 0.05
t_max: 3600
seed: 42

# Rewards
w_wait: 1.0
w_incar: 0.2
r_alight: 0.1
r_board: 0.02
penalty_normalize: true

# Adaptive Rewards (optional)
use_adaptive_reward: false
baseline_config: "simple"
baseline_weight: 0.5
performance_bonus_scale: 1.0
comparative_penalty_scale: 2.0
curriculum_stage: 0
use_dynamic_weights: false

# Agent Hyperparameters
gamma: 0.99
lr: 0.0001
batch_size: 64
replay_capacity: 200000
target_update_steps: 10000
tau: 1.0
epsilon_start: 1.0
epsilon_end: 0.05
decay_steps: 200000
grad_clip: 5.0

# Architecture
dueling: true
use_per: false
per_alpha: 0.6
per_beta: 0.4
per_beta_increment: 0.0
use_vdn: true
use_central_bias: false

# Training
training_steps: 100000
warmup_steps: 64
log_interval: 1000
ckpt_interval: 50000
logdir: dsp2/logs
```

## Design Patterns & Best Practices

### 1. Action Masking
**Problem:** Agent can select invalid actions (up at top floor)
**Solution:** Binary mask filters Q-values before argmax
```python
Q[~mask] = -inf
action = argmax(Q)
```

### 2. Reward Scaling
**Problem:** Large negative rewards during learning destabilize training
**Solutions:**
- Penalty capping (max 30 waiting, 15 in-car)
- Optional normalization by capacity
- Positive shaping rewards (boarding, alighting)

### 3. Curriculum Learning
**Problem:** Full problem too hard to learn from scratch
**Solution:** Progressive difficulty through algorithm targets
```
Stage 0: Beat Random
Stage 1: Beat Collective
Stage 2: Beat Nearest Car
Stage 3: Beat Sectoring
```

### 4. Centralized Training, Decentralized Execution (CTDE)
**Problem:** Independent elevator learning ignores coordination
**Solution:** VDN mixer combines individual Q-values
- Training: Uses global state for mixing
- Execution: Each elevator acts on local observation

### 5. Experience Replay
**Problem:** Online learning from correlated sequential data
**Solution:** Replay buffer breaks correlations
- Uniform sampling (standard)
- Prioritized sampling (PER)

### 6. Target Network
**Problem:** Moving target in TD learning causes instability
**Solution:** Delayed copy of policy network
- Hard update every N steps
- Soft update with tau < 1.0

## Performance Metrics

### Primary Metrics
1. **Average Waiting Time (AWT)**
   - Time passengers spend waiting in hall queues
   - Lower is better
   - Typical range: 2-30 time steps

2. **Average Journey Time (AJT)**
   - Time from boarding to alighting
   - Lower is better
   - Typical range: 3-20 time steps

3. **Episode Return**
   - Cumulative reward over episode
   - Higher is better (less negative)
   - Typical range: -5000 to -1000

### Secondary Metrics
4. **Average Queue Size** - Mean passengers waiting per step
5. **Average In-Car Count** - Mean passengers in elevators per step
6. **Service Rate** - Percentage of passengers successfully served
7. **System Time** - AWT + AJT

## Troubleshooting

### Issue: Agent learns slowly or not at all

**Diagnosis:**
- Check epsilon schedule (enough exploration?)
- Verify reward scale (too large/small?)
- Inspect Q-values (collapsing to single value?)

**Solutions:**
- Increase `decay_steps` for longer exploration
- Adjust `w_wait`, `w_incar` weights
- Enable dueling architecture
- Reduce learning rate if unstable

### Issue: High variance in returns

**Diagnosis:**
- Traffic arrival rate too high?
- Reward penalties too harsh?
- Insufficient training steps?

**Solutions:**
- Reduce `lambda` for easier traffic
- Enable penalty capping
- Increase `training_steps`
- Use larger replay buffer

### Issue: Agent worse than random

**Diagnosis:**
- Reward structure misaligned?
- Network not learning?
- Action masking broken?

**Solutions:**
- Verify positive rewards for good actions
- Check gradient flow (grad_clip)
- Test mask function separately
- Use simpler config (quick.yaml)

### Issue: Training unstable (NaN loss)

**Diagnosis:**
- Exploding gradients?
- Extreme rewards?
- Learning rate too high?

**Solutions:**
- Enable gradient clipping
- Clip rewards to [-10, 10]
- Reduce learning rate
- Disable adaptive rewards temporarily

## Future Extensions

1. **Advanced MARL**
   - QMIX (non-linear mixing)
   - QTRAN (full factorization)
   - MAPPO (policy gradient)

2. **State Representation**
   - Graph neural networks for elevator topology
   - Attention mechanisms for passenger priorities
   - Recurrent networks for time series

3. **Traffic Patterns**
   - Morning/evening rush hours
   - Lunch patterns
   - Inter-floor traffic analysis

4. **Multi-Objective RL**
   - Pareto-optimal solutions
   - Constraint satisfaction
   - Energy efficiency

5. **Sim-to-Real**
   - Domain randomization
   - System identification
   - Hardware deployment

## References

- **DDQN**: [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
- **Dueling DQN**: [Dueling Network Architectures](https://arxiv.org/abs/1511.06581)
- **VDN**: [Value-Decomposition Networks](https://arxiv.org/abs/1706.05296)
- **PER**: [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
