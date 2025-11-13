# Quick Start Guide

Get started with DSP2 in 5 minutes!

## Table of Contents
- [Installation](#installation)
- [First Training Run](#first-training-run)
- [Evaluating Performance](#evaluating-performance)
- [Understanding Results](#understanding-results)
- [Next Steps](#next-steps)

---

## Installation

### 1. Clone and Setup Environment

```bash
# Navigate to project directory
cd DSP391m

# Create virtual environment (recommended)
python -m venv .venv

# Activate environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation

```python
# Test import
python -c "from dsp2.env.egcs_env import EGCSEnv; print('✅ Installation successful!')"
```

---

## First Training Run

### Option 1: Quick Test (2-3 minutes)

Train on a minimal configuration for quick verification:

```bash
python run_training.py
```

This uses the `quick.yaml` config by default:
- 3 floors, 1 elevator
- 2,000 training steps
- Low traffic (λ=0.05)
- Results in `dsp2/logs/`

### Option 2: Simple Training (10-15 minutes)

More realistic but still fast:

```python
# Edit run_training.py to use simple config
# Change line: config_path = os.path.join('dsp2', 'configs', 'simple.yaml')
python run_training.py
```

Simple config:
- 5 floors, 2 elevators
- 10,000 training steps
- Moderate traffic

### Option 3: Full Training (1-2 hours)

Production-quality training:

```python
# Edit run_training.py to use default config
# Change line: config_path = os.path.join('dsp2', 'configs', 'default.yaml')
python run_training.py
```

Default config:
- 10 floors, 2 elevators
- 100,000 training steps
- Higher traffic

---

## Training Output

During training, you'll see output like:

```
================================================================================
TRAINING WITH UPDATED HYPERPARAMETERS
================================================================================
Environment: 5 floors, 2 elevators, capacity=6
Lambda (arrival rate): 0.05

Step 1000/10000 | eps=0.950 | ep=5 | waiting=3 | ep_return=-1234.56
Step 2000/10000 | eps=0.900 | ep=12 | waiting=2 | ep_return=-987.23
...
Saved checkpoint: dsp2/logs/ckpt_step_10000.pt
Training complete! Final checkpoint saved: dsp2/logs/ckpt_final.pt
```

**Key Metrics:**
- `eps`: Exploration rate (decreases over time)
- `ep`: Episode number
- `waiting`: Passengers currently waiting
- `ep_return`: Cumulative reward (higher is better, typically negative)

---

## Evaluating Performance

### 1. Quick Evaluation (Built-in)

The training script automatically evaluates after training:

```
================================================================================
EVALUATION
================================================================================
Episode 1/10: Return=-1234.56, Avg Waiting=2.34, Avg In-Car=1.23
Episode 2/10: Return=-1189.45, Avg Waiting=2.21, Avg In-Car=1.15
...
================================================================================
EVALUATION RESULTS
================================================================================
Mean Return:      -1198.45 ± 45.67
Mean Avg Waiting:     2.28 ± 0.12
Mean Avg In-Car:      1.19 ± 0.08
================================================================================
```

### 2. Compare Against Baselines

Evaluate traditional algorithms to establish baselines:

```bash
python evaluate_traditional.py --config simple --episodes 10
```

This evaluates 5 traditional algorithms:
- Random (lower bound)
- Collective Control
- Nearest Car
- Sectoring
- Fixed Priority

**Output:**
```
================================================================================
TRADITIONAL EGCS ALGORITHMS - PERFORMANCE COMPARISON
================================================================================
Algorithm            Return        AWT        AJT     System      Queue     InCar    Served     Rate
------------------------------------------------------------------------------------------------------------------------
random               -3450.23       8.45      15.23      23.68       4.12      2.34      180.5    99.2%
collective           -2134.56       5.23      12.67      17.90       2.45      1.89      182.3    99.5%
nearest_car          -1567.89       3.12      10.45      13.57       1.34      1.45      183.1    99.7%
sectoring            -1789.34       3.67      11.23      14.90       1.67      1.56      182.8    99.6%
fixed_priority       -2345.67       5.89      13.45      19.34       2.89      2.01      181.7    99.4%
================================================================================

🏆 Best Return:      nearest_car          (-1567.89)
🏆 Best AWT:         nearest_car          (3.12 steps)
🏆 Best AJT:         nearest_car          (10.45 steps)
```

### 3. Final Comprehensive Evaluation

Compare your trained agent vs random baseline:

```bash
python final_evaluation.py
```

**Output:**
```
================================================================================
FINAL EVALUATION - DEFAULT CONFIG
================================================================================
------------------------------------------------------------------------
RANDOM POLICY
------------------------------------------------------------------------
Mean Return:      -3450.23 ± 234.56
Mean Avg Waiting:     8.45
Mean Avg In-Car:      2.34

------------------------------------------------------------------------
TRAINED POLICY
------------------------------------------------------------------------
Mean Return:      -1789.34 ± 123.45
Mean Avg Waiting:     3.67
Mean Avg In-Car:      1.56

================================================================================
PERFORMANCE COMPARISON
================================================================================
Return Improvement:  +1660.89 (absolute)
Waiting Reduction:   +4.78 passengers

✅ SUCCESS: Trained agent is BETTER than random policy!
   The agent achieves 1660.9 higher return.
================================================================================
```

---

## Understanding Results

### Key Performance Metrics

#### 1. Episode Return
- **What**: Cumulative reward over episode
- **Range**: Typically -5000 to -500 (less negative is better)
- **Target**: Beat random baseline by 500+ points

**Interpretation:**
```
Return > -1000: Excellent performance
Return > -2000: Good performance
Return > -3000: Acceptable performance
Return < -3000: Needs improvement
```

#### 2. Average Waiting Time (AWT)
- **What**: Mean time passengers wait in hall queues
- **Unit**: Time steps (typically 1 step = 1 second)
- **Target**: < 5.0 steps (< 5 seconds)

**Interpretation:**
```
AWT < 3.0:  Excellent (matches best traditional algorithms)
AWT < 5.0:  Good
AWT < 8.0:  Acceptable
AWT > 10.0: Poor (passengers waiting too long)
```

#### 3. Average Journey Time (AJT)
- **What**: Mean time from boarding to alighting
- **Unit**: Time steps
- **Target**: < 15.0 steps

**Interpretation:**
```
AJT < 10.0: Excellent
AJT < 15.0: Good
AJT < 20.0: Acceptable
AJT > 25.0: Poor (inefficient routing)
```

#### 4. Service Rate
- **What**: Percentage of passengers successfully served
- **Target**: > 99%

**Interpretation:**
```
Rate > 99.5%: Excellent
Rate > 99.0%: Good
Rate > 98.0%: Acceptable
Rate < 98.0%: Poor (passengers stuck in queues)
```

### Training Progress Indicators

#### 1. Epsilon Decay
- Starts at 1.0 (full exploration)
- Decreases to 0.05 (mostly exploitation)
- Should decrease smoothly over training

#### 2. Episode Return Trend
- Should increase (become less negative) over time
- Variance decreases as learning stabilizes
- Plateau indicates convergence

#### 3. Waiting Passengers
- Should decrease over training
- Stable low values indicate good policy
- Spikes indicate traffic bursts

---

## Visualizing Results with TensorBoard

View training progress in real-time:

```bash
# Start TensorBoard
tensorboard --logdir=dsp2/logs

# Open in browser
# Navigate to: http://localhost:6006
```

**Key Plots to Check:**

1. **Loss** - Should decrease and stabilize
2. **Q-mean** - Should increase (agent learns value)
3. **Episode Return** - Should increase over time
4. **n_waiting** - Should decrease
5. **awt / ajt** - Should decrease (if adaptive rewards enabled)

---

## Troubleshooting

### Issue: Training is too slow

**Solution:**
```yaml
# Use quick.yaml config
# Reduce training_steps
training_steps: 5000  # Instead of 100000
```

### Issue: Agent not learning (return not improving)

**Checklist:**
1. ✓ Is epsilon decaying? (Should reach ~0.1 by mid-training)
2. ✓ Is loss decreasing?
3. ✓ Are Q-values increasing?
4. ✓ Is learning rate appropriate? (Try 1e-3 if 1e-4 too slow)

**Quick Fix:**
```yaml
# In config file, adjust:
lr: 0.001           # Increase from 0.0001
decay_steps: 5000   # Decrease from 200000 (faster exploration->exploitation)
```

### Issue: High variance in results

**Solution:**
```yaml
# Increase batch size and replay capacity
batch_size: 128         # From 64
replay_capacity: 100000 # From 50000
```

### Issue: Agent worse than random

**Diagnosis:**
- Reward structure may be misaligned
- Check if penalties are too harsh
- Verify action masking is working

**Fix:**
```yaml
# Reduce penalty weights
w_wait: 0.5   # From 1.0
w_incar: 0.1  # From 0.2

# Increase positive rewards
r_alight: 1.0  # From 0.1
r_board: 0.5   # From 0.02
```

### Issue: Import errors

```bash
# Ensure project root is in path
export PYTHONPATH="${PYTHONPATH}:/path/to/DSP391m"

# Or run as module
python -m dsp2.train.train
```

---

## Next Steps

### 1. Experiment with Configurations

Try different environment settings:

```yaml
# dsp2/configs/my_config.yaml
n_floors: 8          # Try different building sizes
m_elevators: 3       # More elevators
capacity: 10         # Larger elevators
lambda: 0.1          # Higher traffic
```

### 2. Enable Adaptive Rewards

Unlock curriculum learning and performance bonuses:

```yaml
# In config file
use_adaptive_reward: true
baseline_config: "simple"
curriculum_stage: 0  # Start at stage 0 (beat random)
use_dynamic_weights: true
```

Then run:
```bash
# First, generate baselines
python evaluate_traditional.py --config simple

# Then train with adaptive rewards
python run_training.py  # Using adaptive.yaml
```

### 3. Tune Hyperparameters

Experiment with agent settings:

```yaml
# Learning
lr: 0.0005           # Try different learning rates
gamma: 0.95          # Less future discounting
batch_size: 128      # Larger batches

# Exploration
epsilon_start: 1.0
epsilon_end: 0.01    # More exploitation
decay_steps: 50000   # Faster decay

# Architecture
dueling: true        # Enable dueling DQN
use_vdn: false       # Disable multi-agent mixing for simplicity
```

### 4. Implement Custom Algorithms

Create your own dispatcher:

```python
# dsp2/agents/my_algorithm.py
class MyCustomDispatcher:
    def __init__(self, n_floors, m_elevators, capacity):
        self.N = n_floors
        self.M = m_elevators
        
    def select_action(self, state, positions, directions, 
                     hall_up, hall_down, car_calls, onboard_counts):
        # Your logic here
        actions = np.zeros(self.M, dtype=int)
        # ... implement your algorithm
        return actions
```

### 5. Analyze Agent Behavior

Add visualization and debugging:

```python
# Visualize Q-values
state = env.reset()
mask = env.action_mask()
Q = agent.policy(torch.from_numpy(state).float().unsqueeze(0))
print("Q-values:", Q)

# Track action distribution
action_counts = np.zeros(4)
for _ in range(100):
    action = agent.select_action(state, mask, epsilon=0.0)
    for a in action:
        action_counts[a] += 1
print("Action distribution:", action_counts / action_counts.sum())
```

### 6. Multi-Scenario Testing

Test robustness across different traffic patterns:

```python
# Morning rush (ground floor arrivals)
lambda_fn = lambda t: 0.1 if t < 600 else 0.02

# Evening rush (return to ground)
# Modify passenger destination distribution

# Lunch traffic (inter-floor)
# Random source-destination pairs
```

---

## Common Workflows

### Workflow 1: Quick Development Cycle

```bash
# 1. Modify code
vim dsp2/agents/ddqn_agent.py

# 2. Quick test
python run_training.py  # Uses quick.yaml by default

# 3. Check TensorBoard
tensorboard --logdir=dsp2/logs

# 4. Iterate
```

### Workflow 2: Hyperparameter Search

```bash
# Generate baselines once
python evaluate_traditional.py --config simple

# Try different configs
for config in simple mini default adaptive; do
    python run_training.py --config $config
    python final_evaluation.py --config $config
done
```

### Workflow 3: Benchmark Study

```bash
# 1. Evaluate all traditional algorithms
python evaluate_traditional.py --config default --episodes 20

# 2. Train RL agent
python run_training.py --config default

# 3. Compare results
python final_evaluation.py --config default

# 4. Generate report
python -m dsp2.visualization.compare_algorithms
```

---

## Best Practices

### 1. Start Small, Scale Up
- Begin with `quick.yaml` to verify setup
- Move to `simple.yaml` for development
- Use `default.yaml` for final training

### 2. Monitor Training
- Always check TensorBoard
- Look for smooth loss curves
- Verify Q-values are increasing

### 3. Checkpoint Regularly
```yaml
ckpt_interval: 10000  # Save every 10K steps
```

### 4. Reproducibility
```yaml
seed: 42  # Always set seed
```
```python
np.random.seed(42)
torch.manual_seed(42)
```

### 5. Compare Against Baselines
- Always evaluate traditional algorithms first
- Use adaptive rewards with baseline comparison
- Track improvement over random

---

## Resources

### Documentation
- [Architecture Documentation](ARCHITECTURE.md) - Detailed system design
- [API Reference](API_REFERENCE.md) - Complete API docs
- [README](../README.md) - Project overview

### Configuration Examples
- `dsp2/configs/quick.yaml` - Fast testing
- `dsp2/configs/simple.yaml` - Development
- `dsp2/configs/default.yaml` - Production
- `dsp2/configs/adaptive.yaml` - With adaptive rewards

### Example Scripts
- `run_training.py` - Main training script
- `evaluate_traditional.py` - Baseline evaluation
- `final_evaluation.py` - Agent vs baseline comparison

---

## Getting Help

### Common Questions

**Q: How long should I train?**
A: Depends on config:
- Quick: 2K-5K steps (2-3 min)
- Simple: 10K-20K steps (10-15 min)
- Default: 100K-200K steps (1-2 hours)

**Q: What's a good return value?**
A: Compare to baselines:
- Beat random: Good start
- Beat collective: Decent
- Beat nearest_car: Excellent
- Beat all: State-of-the-art

**Q: Why is training unstable?**
A: Try:
- Reduce learning rate
- Increase batch size
- Enable gradient clipping
- Use reward clipping

**Q: Should I use adaptive rewards?**
A: Yes if:
- You have baseline metrics
- Want curriculum learning
- Need better sample efficiency

No if:
- First time training
- Debugging basic setup
- Comparing raw RL approaches

---

## Next: Dive Deeper

Once comfortable with basics, explore:
1. [Architecture Documentation](ARCHITECTURE.md) - System internals
2. [API Reference](API_REFERENCE.md) - Detailed API
3. Traditional algorithm implementation
4. Custom reward functions
5. Multi-agent coordination
6. Real-world traffic patterns

Happy training! 🚀
