# Reward Model Documentation

Comprehensive guide to the reward system in DSP2.

## Table of Contents
- [Overview](#overview)
- [Base Reward Structure](#base-reward-structure)
- [Adaptive Reward System](#adaptive-reward-system)
- [Reward Engineering Techniques](#reward-engineering-techniques)
- [Tuning Guide](#tuning-guide)
- [Examples](#examples)

---

## Overview

The reward system is the core mechanism for training the RL agent. DSP2 implements a sophisticated multi-component reward structure that combines:

1. **Base Penalties** - Discourage queue buildup and delays
2. **Positive Rewards** - Encourage beneficial actions (boarding, alighting)
3. **Adaptive Bonuses** - Compare performance against baselines
4. **Dynamic Weights** - Adjust penalties based on current performance

### Design Philosophy

**Key Principle:** Reward should reflect real-world elevator system objectives:
- Minimize passenger waiting time
- Minimize passenger journey time  
- Maximize throughput
- Balance load across elevators

**Challenge:** Raw penalties can cause training instability due to:
- Quadratic growth with traffic (more arrivals → longer queues → higher penalties)
- Catastrophic failures during exploration (agent ignores calls → queues explode)
- Reward variance across episodes with different traffic

**Solution:** Multi-layered approach with:
- Penalty capping to prevent catastrophic negative rewards
- Positive shaping to encourage good actions
- Adaptive comparison against known baselines
- Dynamic weight adjustment based on performance

---

## Base Reward Structure

### Formula

```python
reward = -(w_wait * n_waiting_capped + w_incar * n_incar_capped) 
         + r_alight * passengers_alighted 
         + r_board * passengers_boarded
```

### Components

#### 1. Waiting Penalty

**Purpose:** Discourage passengers waiting in hall queues

```python
waiting_penalty = w_wait * min(n_waiting, MAX_WAITING)
```

**Parameters:**
- `w_wait` (default: 1.0) - Weight for waiting penalty
- `MAX_WAITING` (default: 30) - Cap to prevent catastrophic penalties
- `n_waiting` - Current number of passengers in hall queues

**Rationale:**
- Waiting is the primary source of passenger dissatisfaction
- Capped at 30 to prevent penalty explosion during early training
- Lower cap with higher weight forces agent to act earlier

**Example:**
```
5 passengers waiting:  penalty = -1.0 * 5 = -5.0
20 passengers waiting: penalty = -1.0 * 20 = -20.0
50 passengers waiting: penalty = -1.0 * 30 = -30.0 (capped)
```

#### 2. In-Car Penalty

**Purpose:** Encourage quick delivery to destination

```python
incar_penalty = w_incar * min(n_incar, MAX_INCAR)
```

**Parameters:**
- `w_incar` (default: 0.2) - Weight for in-car penalty (lower than waiting)
- `MAX_INCAR` (default: 15) - Cap for in-car passengers
- `n_incar` - Current number of passengers in elevators

**Rationale:**
- Passengers in elevators are already being served (better than waiting)
- Lower weight than waiting penalty reflects this priority
- Still penalized to encourage efficient routing

**Example:**
```
3 passengers in car: penalty = -0.2 * 3 = -0.6
10 passengers in car: penalty = -0.2 * 10 = -2.0
20 passengers in car: penalty = -0.2 * 15 = -3.0 (capped)
```

#### 3. Alighting Reward

**Purpose:** Positive reinforcement for successful delivery

```python
alight_reward = r_alight * passengers_alighted
```

**Parameters:**
- `r_alight` (default: 0.1) - Reward per passenger delivered
- `passengers_alighted` - Count of passengers reaching destination this step

**Rationale:**
- Provides immediate positive feedback for good actions
- Sparse reward (only when destination reached)
- Small magnitude to avoid overwhelming penalties

**Example:**
```
2 passengers reach destination: reward = +0.1 * 2 = +0.2
5 passengers reach destination: reward = +0.1 * 5 = +0.5
```

#### 4. Boarding Reward

**Purpose:** Encourage picking up waiting passengers

```python
board_reward = r_board * passengers_boarded
```

**Parameters:**
- `r_board` (default: 0.02) - Reward per passenger picked up
- `passengers_boarded` - Count of passengers entering elevator this step

**Rationale:**
- Immediate positive feedback for addressing hall calls
- Lower than alighting (boarding is intermediate step)
- Helps agent learn to open doors at appropriate floors

**Example:**
```
3 passengers board: reward = +0.02 * 3 = +0.06
8 passengers board: reward = +0.02 * 8 = +0.16
```

### Penalty Normalization

Optional feature to make rewards scale-invariant:

```python
if penalty_normalize:
    norm_denom = max(1, n_floors * capacity)
    waiting_penalty /= norm_denom
    incar_penalty /= norm_denom
```

**When to Use:**
- Comparing across different building sizes
- Transfer learning between configurations
- Theoretical analysis of convergence

**When NOT to Use:**
- Single fixed configuration
- Need strong penalty signals for learning
- Debugging reward structure

**Note:** Can make penalties 100x weaker! For a 10-floor, 8-capacity system:
- Without normalization: 30 waiting → penalty = -30
- With normalization: 30 waiting → penalty = -30/80 = -0.375

### Typical Reward Ranges

For default configuration (10 floors, 2 elevators, λ=0.05):

| Scenario | Waiting | In-Car | Base Reward | Notes |
|----------|---------|--------|-------------|-------|
| Idle (no passengers) | 0 | 0 | 0.0 | Neutral |
| Light traffic | 1-5 | 1-3 | -1 to -6 | Normal operation |
| Moderate traffic | 5-15 | 3-8 | -6 to -17 | Needs attention |
| Heavy traffic | 15-30 | 8-15 | -17 to -33 | Stressed system |
| Overload (capped) | 30+ | 15+ | -33.0 | Agent must act! |

**Per Episode:**
- Good performance: -500 to -1000 return
- Average performance: -1000 to -2000 return
- Poor performance: -2000 to -5000 return

---

## Adaptive Reward System

Extends base rewards with performance-based bonuses/penalties by comparing against traditional algorithm baselines.

### Activation

```yaml
# In config file
use_adaptive_reward: true
baseline_config: "simple"  # Which baseline to load
baseline_weight: 0.5       # How much to weight adaptive components
performance_bonus_scale: 1.0
comparative_penalty_scale: 2.0
curriculum_stage: 0        # 0=random, 1=collective, 2=nearest, 3=sectoring
use_dynamic_weights: false
```

### Five Strategies

#### Strategy 1: Dynamic Baseline Thresholding

Graduated bonuses based on performance tier.

```python
if awt <= excellent_threshold:
    bonus = +2.0 * performance_bonus_scale
elif awt <= good_threshold:
    bonus = +1.0 * performance_bonus_scale
elif awt <= average_threshold:
    bonus = 0.0
else:
    bonus = -1.0 * performance_bonus_scale
```

**Thresholds (from baselines):**
- Excellent: Beat best traditional algorithm (e.g., AWT < 2.8)
- Good: Better than mean (e.g., AWT < 3.5)
- Average: Near mean (e.g., AWT < 4.5)
- Poor: Worse than mean (e.g., AWT > 4.5)

**Example:**
```
AWT = 2.5 (excellent): bonus = +2.0
AWT = 3.2 (good):      bonus = +1.0  
AWT = 4.0 (average):   bonus = 0.0
AWT = 6.0 (poor):      bonus = -1.0
```

#### Strategy 2: Comparative Reward Shaping

Smooth comparison against baseline mean using tanh.

```python
awt_ratio = current_awt / baseline_mean_awt
awt_bonus = tanh(1.0 - awt_ratio) * performance_bonus_scale
```

**Properties:**
- Bounded to [-1, 1] by tanh
- Smooth gradients (better than step functions)
- Symmetric around baseline mean

**Example:**
```
current_awt = 2.0, mean_awt = 4.0:
  ratio = 0.5, bonus = tanh(0.5) ≈ +0.46

current_awt = 4.0, mean_awt = 4.0:
  ratio = 1.0, bonus = tanh(0.0) = 0.0

current_awt = 8.0, mean_awt = 4.0:
  ratio = 2.0, bonus = tanh(-1.0) ≈ -0.76
```

#### Strategy 3: Multi-Metric Performance Index

Combines AWT, AJT, and service rate into single score.

```python
awt_score = tanh(1.0 - awt / mean_awt)
ajt_score = tanh(1.0 - ajt / mean_ajt)  
service_score = (service_rate - 0.99)

combined = (awt_score * 0.2 + ajt_score * 0.1 + service_score * 0.05)
bonus = clip(combined * performance_bonus_scale, -1.0, 1.0)
```

**Weights:**
- AWT: 0.2 (primary metric)
- AJT: 0.1 (secondary)
- Service: 0.05 (maintenance)

**Example:**
```
Good agent: AWT=2.0, AJT=10.0, Service=99.5%
  → awt_score ≈ +0.46, ajt_score ≈ +0.38, service_score = +0.005
  → combined ≈ 0.13 → bonus ≈ +0.13

Poor agent: AWT=8.0, AJT=20.0, Service=98.5%
  → awt_score ≈ -0.76, ajt_score ≈ -0.62, service_score = -0.005
  → combined ≈ -0.21 → bonus ≈ -0.21
```

#### Strategy 4: Algorithm-Specific Competitive Rewards

Bonus for beating specific traditional algorithms.

```python
algorithms_beaten = 0
for algo in [random, collective, nearest_car, sectoring, fixed_priority]:
    if current_awt <= algo.awt:
        algorithms_beaten += 1

beat_ratio = algorithms_beaten / 5

if beat_ratio == 1.0:
    bonus = +3.0  # Beat all!
elif beat_ratio >= 0.75:
    bonus = +2.0  # Beat most
elif beat_ratio >= 0.5:
    bonus = +1.0  # Beat half
elif beat_ratio >= 0.25:
    bonus = +0.5  # Beat some
else:
    bonus = 0.0   # Beat none
```

**Example:**
```
AWT = 2.5 beats [random(8.5), collective(5.2), nearest(3.1)]
  → algorithms_beaten = 3/5 = 60%
  → bonus = +1.0

AWT = 2.0 beats all 5 algorithms
  → algorithms_beaten = 5/5 = 100%
  → bonus = +3.0
```

#### Strategy 5: Staged Difficulty Training (Curriculum)

Progressive targets from easiest to hardest algorithm.

```python
curriculum_targets = [
    (stage=0, 'random',      awt=8.5),
    (stage=1, 'collective',  awt=5.2),
    (stage=2, 'nearest_car', awt=3.1),
    (stage=3, 'sectoring',   awt=2.8)
]

target_awt = curriculum_targets[curriculum_stage].awt

if current_awt <= target_awt:
    improvement = (target_awt - current_awt) / target_awt
    bonus = min(improvement, 1.0) * performance_bonus_scale
else:
    deficit_ratio = (current_awt - target_awt) / target_awt
    penalty = -min(deficit_ratio * 0.5, 1.0) * comparative_penalty_scale
```

**Progression:**
```
Stage 0: Beat Random (AWT < 8.5)
  → Easy, agent should achieve quickly
  
Stage 1: Beat Collective (AWT < 5.2)  
  → Moderate, requires coordination

Stage 2: Beat Nearest Car (AWT < 3.1)
  → Hard, requires optimization

Stage 3: Beat Sectoring (AWT < 2.8)
  → Very hard, state-of-the-art performance
```

**Example:**
```
Stage 1 (target=5.2):
  AWT=4.5: bonus = (5.2-4.5)/5.2 ≈ +0.13
  AWT=6.0: penalty = -min(0.15*0.5, 1.0) = -0.075

Stage 3 (target=2.8):
  AWT=2.5: bonus = (2.8-2.5)/2.8 ≈ +0.11
  AWT=4.0: penalty = -min(0.43*0.5, 1.0) = -0.21
```

### Combined Adaptive Reward

All strategies are combined with reduced weights:

```python
adaptive_bonus = (
    tier_bonus * 0.2 +
    comparative * 0.3 +
    multi_metric * 0.1 +
    competitive * 0.1 +
    curriculum * 0.3
)

# Clip to prevent explosion
adaptive_bonus = clip(adaptive_bonus, -5.0, 5.0)

# Final reward
total_reward = base_reward + adaptive_bonus
total_reward = clip(total_reward, -10.0, 10.0)
```

**Weights Rationale:**
- Curriculum (0.3): Primary guidance for staged learning
- Comparative (0.3): Continuous smooth feedback
- Tier (0.2): Discrete milestones
- Multi-metric (0.1): Holistic performance
- Competitive (0.1): Motivation bonus

### Dynamic Weight Adjustment

Automatically adjusts penalty weights based on performance:

```python
if use_dynamic_weights:
    if awt <= excellent_threshold:
        w_wait, w_incar = 0.5, 0.1   # Reduce (trust agent)
    elif awt <= good_threshold:
        w_wait, w_incar = 1.0, 0.2   # Standard
    elif awt <= average_threshold:
        w_wait, w_incar = 1.5, 0.3   # Increase
    else:
        w_wait, w_incar = 2.0, 0.5   # Strong guidance
```

**Effect:**
- Good performance → Lower penalties → More freedom to explore
- Poor performance → Higher penalties → Stronger corrective signal

---

## Reward Engineering Techniques

### 1. Penalty Capping

**Problem:** Uncapped penalties grow quadratically with traffic
- 10 waiting → -10
- 50 waiting → -50
- 200 waiting → -200 (catastrophic)

**Solution:** Cap at reasonable maximum
```python
MAX_WAITING = 30
MAX_INCAR = 15

waiting_penalty = w_wait * min(n_waiting, MAX_WAITING)
incar_penalty = w_incar * min(n_incar, MAX_INCAR)
```

**Benefits:**
- Prevents reward explosion during exploration
- Bounds gradient magnitudes for stable learning
- Agent learns "avoid reaching cap" instead of "minimize absolute count"

**Tuning Tips:**
- Lower cap + higher weight → Early intervention
- Higher cap + lower weight → More tolerance
- Cap at 50 is too lenient (agent lazy until 40-50 waiting)
- Cap at 30 with w_wait=0.4 → Strong signal at 20-30 waiting

### 2. Reward Shaping

**Problem:** Pure penalty is dense but negative; hard to learn good actions

**Solution:** Add positive shaping rewards
```python
reward += r_alight * passengers_alighted
reward += r_board * passengers_boarded
```

**Benefits:**
- Provides immediate positive feedback
- Helps credit assignment (which action was good?)
- Breaks symmetry (all actions equally bad → some actions good)

**Tuning Tips:**
- Alighting > Boarding (delivery more important than pickup)
- Keep smaller than penalties (penalties should dominate)
- Too large → Agent ignores penalties, optimizes only shaping
- Typical ratio: r_alight/w_wait ≈ 0.1 to 0.5

### 3. Reward Clipping

**Problem:** Extreme rewards during rare events cause instability

**Solution:** Clip total reward to reasonable range
```python
final_reward = np.clip(reward, -10.0, 10.0)
```

**Benefits:**
- Prevents single-step dominance in bellman updates
- Stabilizes gradient magnitudes
- Makes learning less sensitive to outliers

**Tuning Tips:**
- Clip after all components computed
- Range should contain 99% of typical rewards
- Too tight → Information loss
- Too loose → No effect

### 4. Reward Scaling

**Problem:** Absolute reward magnitude affects learning rate sensitivity

**Solution A:** Scale rewards to standard range [-1, 1]
```python
reward = reward / expected_max_magnitude
```

**Solution B:** Use reward normalization
```python
reward = (reward - running_mean) / (running_std + eps)
```

**Benefits:**
- Makes learning rate tuning more robust
- Enables transfer between configurations
- Consistent gradient magnitudes

**Tuning Tips:**
- Not always necessary with proper hyperparameters
- Can make debugging harder (raw rewards obscured)
- Useful for very sparse or very dense rewards

### 5. Curriculum Reward Adjustment

**Problem:** Full reward function too complex to learn from scratch

**Solution:** Progressive difficulty
```python
# Stage 1: Simple reward (only waiting)
reward = -w_wait * n_waiting

# Stage 2: Add in-car penalty
reward = -w_wait * n_waiting - w_incar * n_incar

# Stage 3: Add shaping
reward = -w_wait * n_waiting - w_incar * n_incar + r_alight * alighted

# Stage 4: Full reward with adaptive bonuses
reward = base_reward + adaptive_bonus
```

**Benefits:**
- Easier to learn basic behaviors first
- Prevents early catastrophic failure
- Smoother learning curve

---

## Tuning Guide

### Step-by-Step Reward Tuning

#### Step 1: Establish Baseline

Run random policy to understand reward range:

```python
env = EGCSEnv(n_floors=10, m_elevators=2)
total_reward = 0
state = env.reset()
done = False

while not done:
    action = np.random.randint(0, 4, size=2)
    state, reward, done, info = env.step(action)
    total_reward += reward
    
print(f"Random policy return: {total_reward}")
# Typical: -3000 to -5000 for default config
```

#### Step 2: Tune Penalty Weights

Goal: Penalties should dominate but not overwhelm

```python
# Test different weights
for w_wait in [0.5, 1.0, 1.5, 2.0]:
    for w_incar in [0.1, 0.2, 0.3, 0.5]:
        env = EGCSEnv(w_wait=w_wait, w_incar=w_incar)
        # Run evaluation
        # Check: waiting time, in-car time, return
```

**Guidelines:**
- Start with w_wait=1.0, w_incar=0.2 (waiting 5x more important)
- If agent ignores waiting queues → increase w_wait
- If agent never opens doors → increase r_alight, r_board
- If agent too conservative → reduce penalties

**Target Ratios:**
```
w_wait : w_incar : r_alight : r_board
  1.0  :   0.2   :   0.1    :  0.02
  (or)
  2.0  :   0.5   :   3.0    :  1.5
```

#### Step 3: Tune Penalty Caps

Goal: Prevent catastrophic penalties while maintaining signal

```python
# Analyze queue sizes during training
max_waiting_seen = []
for episode in training:
    max_waiting_seen.append(max(info['n_waiting'] for info in episode))

# Set cap at 95th percentile
MAX_WAITING = np.percentile(max_waiting_seen, 95)
```

**Guidelines:**
- Too low → Agent can't differentiate 30 vs 50 waiting (information loss)
- Too high → Large penalties destabilize training
- Start at 30 (waiting) and 15 (in-car) for 8-capacity elevators
- Scale with capacity: MAX_WAITING ≈ 3-4 × capacity × n_elevators

#### Step 4: Tune Shaping Rewards

Goal: Provide positive signal without overwhelming penalties

```python
# Measure typical counts per episode
alighted_per_episode = []
boarded_per_episode = []
# ... collect data ...

# Shaping should contribute 5-10% of total reward magnitude
avg_alighted = np.mean(alighted_per_episode)  # e.g., 180
total_penalty_magnitude = 2000  # from baseline

r_alight = (0.05 * total_penalty_magnitude) / avg_alighted
# r_alight ≈ (0.05 * 2000) / 180 ≈ 0.56
```

**Guidelines:**
- Alighting more important than boarding: r_alight ≈ 5-10× r_board
- Total positive rewards should be 5-10% of total penalties
- If agent "milks" rewards (opens/closes repeatedly) → reduce shaping
- If agent ignores passengers → increase shaping

#### Step 5: Enable Adaptive Rewards (Optional)

After base rewards are working:

```python
# 1. Generate baselines
python evaluate_traditional.py --config simple --episodes 20

# 2. Enable adaptive rewards
use_adaptive_reward: true
baseline_config: "simple"
baseline_weight: 0.3  # Start conservative

# 3. Monitor performance tier in TensorBoard
# 4. Gradually increase baseline_weight to 0.5-0.7
```

**Guidelines:**
- Start with baseline_weight=0.2-0.3 (don't disrupt base learning)
- Increase if training is stable and improving
- Reduce if training becomes unstable
- Use curriculum_stage for progressive difficulty

### Debugging Reward Issues

#### Issue: Agent not learning

**Symptoms:**
- Return not improving over time
- Q-values not increasing
- Random-like behavior after training

**Diagnosis:**
```python
# Check reward variance
rewards = [step_reward for episode in training for step_reward in episode]
print(f"Reward mean: {np.mean(rewards)}")
print(f"Reward std: {np.std(rewards)}")
print(f"Reward range: [{np.min(rewards)}, {np.max(rewards)}]")

# Check if penalties dominate
print(f"Positive rewards: {sum(r for r in rewards if r > 0)}")
print(f"Negative rewards: {sum(r for r in rewards if r < 0)}")
```

**Solutions:**
- If high variance → Clip rewards, increase batch size
- If always negative → Add more shaping rewards
- If range too large → Adjust penalty weights

#### Issue: Agent too conservative (doesn't pick up passengers)

**Symptoms:**
- Low boarding counts
- Doors rarely open
- Waiting queues grow but agent ignores

**Solutions:**
```yaml
r_board: 1.0      # Increase from 0.02
r_alight: 3.0     # Increase from 0.1
w_wait: 2.0       # Increase penalty pressure
```

#### Issue: Agent too aggressive (opens doors unnecessarily)

**Symptoms:**
- High action=3 (open) frequency
- Opens at floors with no passengers
- "Milks" rewards by repeated open/close

**Solutions:**
```yaml
r_board: 0.01     # Decrease shaping
r_alight: 0.05    # Decrease shaping
w_wait: 0.5       # Reduce penalty pressure
```

#### Issue: Reward explosion (NaN loss)

**Symptoms:**
- Loss becomes NaN
- Q-values explode to infinity
- Gradients are very large

**Solutions:**
```python
# Add reward clipping
reward = np.clip(reward, -10.0, 10.0)

# Add gradient clipping
grad_clip: 5.0

# Reduce learning rate
lr: 0.00005
```

---

## Examples

### Example 1: Simple Penalty-Only Reward

Minimalist approach for debugging:

```python
env = EGCSEnv(
    n_floors=5,
    m_elevators=2,
    capacity=4,
    w_wait=1.0,
    w_incar=0.0,    # Ignore in-car
    r_alight=0.0,   # No shaping
    r_board=0.0,    # No shaping
    penalty_normalize=False
)
```

**Use Case:**
- Understanding basic penalty dynamics
- Debugging environment
- Testing if agent can learn anything

**Expected Behavior:**
- Agent learns to reduce waiting queues
- May not optimize journey time (no in-car penalty)
- Sparse learning signal (only queue reduction matters)

### Example 2: Balanced Penalty + Shaping

Recommended starting point:

```python
env = EGCSEnv(
    n_floors=10,
    m_elevators=2,
    capacity=8,
    w_wait=1.0,
    w_incar=0.2,
    r_alight=0.1,
    r_board=0.02,
    penalty_normalize=False
)
```

**Use Case:**
- General purpose training
- Good balance of signals
- Proven to work across scenarios

**Expected Behavior:**
- Agent learns to reduce waiting queues (primary)
- Also optimizes journey time (secondary)
- Positive feedback helps credit assignment

### Example 3: Aggressive Shaping

For sparse reward environments:

```python
env = EGCSEnv(
    n_floors=10,
    m_elevators=2,
    capacity=8,
    w_wait=1.0,
    w_incar=0.5,
    r_alight=3.0,    # 30x standard
    r_board=1.5,     # 75x standard
    penalty_normalize=False
)
```

**Use Case:**
- Low traffic scenarios (few passengers)
- Need stronger positive signal
- Faster initial learning

**Expected Behavior:**
- Agent quickly learns to pick up and deliver
- May overfit to maximizing board/alight count
- Risk of reward hacking (unnecessary opens)

### Example 4: Adaptive Rewards with Curriculum

Full system with progressive difficulty:

```python
env = EGCSEnv(
    n_floors=10,
    m_elevators=2,
    capacity=8,
    w_wait=1.0,
    w_incar=0.2,
    r_alight=0.1,
    r_board=0.02,
    penalty_normalize=False,
    # Adaptive rewards
    use_adaptive_reward=True,
    baseline_config='simple',
    baseline_weight=0.5,
    performance_bonus_scale=1.0,
    comparative_penalty_scale=2.0,
    curriculum_stage=0,  # Start at beat random
    use_dynamic_weights=True
)
```

**Use Case:**
- Production training
- Maximum sample efficiency
- Progressive improvement goals

**Expected Behavior:**
- Stage 0: Learns to beat random (baseline)
- Stage 1: Improves to beat collective control
- Stage 2: Optimizes to beat nearest car
- Stage 3: Reaches state-of-the-art (beats sectoring)

### Example 5: Normalized Rewards for Transfer Learning

Scale-invariant rewards:

```python
env = EGCSEnv(
    n_floors=10,
    m_elevators=2,
    capacity=8,
    w_wait=10.0,     # Higher weight
    w_incar=2.0,     # Higher weight
    r_alight=1.0,    # Higher reward
    r_board=0.2,     # Higher reward
    penalty_normalize=True  # Enable normalization
)
```

**Use Case:**
- Training across different building sizes
- Theoretical analysis
- Transfer learning experiments

**Expected Behavior:**
- Rewards in consistent range across configs
- Can transfer learned policy between sizes
- May need more training steps (weaker signals)

---

## Summary

### Key Takeaways

1. **Base Rewards:**
   - Penalties for waiting (w_wait=1.0) and in-car (w_incar=0.2)
   - Shaping for alighting (r_alight=0.1) and boarding (r_board=0.02)
   - Capped at 30 waiting, 15 in-car to prevent explosion

2. **Adaptive Rewards:**
   - Five strategies for performance comparison
   - Curriculum learning through staged targets
   - Dynamic weight adjustment based on performance

3. **Tuning Process:**
   - Start with simple penalty-only
   - Add shaping gradually
   - Enable adaptive rewards after base works
   - Monitor TensorBoard for stability

4. **Common Pitfalls:**
   - Uncapped penalties → training instability
   - Too much shaping → reward hacking
   - No shaping → slow learning
   - Wrong penalty ratio → suboptimal priorities

### Best Practices

✅ **DO:**
- Cap penalties to prevent catastrophic failures
- Use shaping rewards for faster learning
- Start simple, add complexity gradually
- Monitor reward distributions during training
- Compare against baselines

❌ **DON'T:**
- Use huge uncapped penalties
- Make shaping rewards larger than penalties
- Enable all features at once without testing
- Ignore reward variance and outliers
- Skip baseline evaluation

### Quick Reference

**Default Good Starting Point:**
```yaml
w_wait: 1.0
w_incar: 0.2
r_alight: 0.1
r_board: 0.02
penalty_normalize: false
use_adaptive_reward: false
```

**For Faster Learning:**
```yaml
w_wait: 1.5
w_incar: 0.3
r_alight: 1.0
r_board: 0.5
```

**For Stable Training:**
```yaml
w_wait: 0.5
w_incar: 0.1
r_alight: 0.05
r_board: 0.01
```

**For Curriculum Learning:**
```yaml
use_adaptive_reward: true
baseline_config: "simple"
curriculum_stage: 0  # Increment as agent improves
```
