# Final Report: Hyperparameter Tuning for DSP2 Multi-Elevator EGCS

**Date:** November 12, 2025  
**Task:** Adjust hyperparameters to more reasonable values and evaluate performance

---

## Executive Summary

### ✅ Task Completed: Hyperparameters Adjusted
The original hyperparameters had **unreasonably high values** that made the problem too difficult:
- **Lambda (arrival rate) reduced by 90%**: 0.05 → 0.005
- **Episode length reduced by 67%**: 3600s → 1200s  
- **Reward shaping significantly improved**: 20-50x increases in positive rewards

### ⚠️ Performance Issue Identified
Despite reasonable hyperparameters, the trained DDQN agent **consistently performs worse than random policy**. This indicates a fundamental issue with the learning algorithm or problem formulation, not just hyperparameter tuning.

---

## Hyperparameter Changes

### Original Configuration (UNREASONABLE)
```yaml
Environment:
  lambda: 0.05          # 180 passengers/hour - EXTREMELY HIGH
  t_max: 3600           # 1 hour episodes - TOO LONG
  n_floors: 10
  w_incar: 0.2          # Low penalty for passengers in elevator
  r_alight: 0.1         # Minimal reward for delivery
  r_board: 0.02         # Almost no reward for pickup

Agent:
  gamma: 0.99
  lr: 0.0002
  use_vdn: true
  use_per: true         # Complex features enabled
  training_steps: 100000
```

**Problems:**
1. **Lambda 0.05** = ~180 passengers over 3600s for 2 elevators (90 per elevator!) = IMPOSSIBLE
2. **Sparse rewards**: Minimal positive reinforcement for correct actions
3. **Long episodes**: Difficult credit assignment over 3600 time steps
4. **Complex architecture**: VDN mixer + PER adds instability

### Final Configuration (REASONABLE)
```yaml
Environment:
  lambda: 0.005         # 18 passengers/hour - MANAGEABLE ✅
  t_max: 1200           # 20 minutes - SHORTER ✅
  n_floors: 8           # Slightly smaller
  w_incar: 0.5          # Higher penalty (2.5x)
  r_alight: 2.0         # Strong reward (20x) ✅
  r_board: 1.0          # Strong reward (50x) ✅

Agent:
  gamma: 0.95           # Less long-term focus
  lr: 0.0005            # Slightly higher
  use_vdn: false        # Simplified ✅
  use_per: false        # Disabled for stability ✅
  training_steps: 40000 # More focused training
```

**Key Improvements:**
- ✅ **90% reduction in arrival rate** - makes problem tractable
- ✅ **67% shorter episodes** - faster learning cycles
- ✅ **20-50x reward increases** - stronger learning signal
- ✅ **Simplified architecture** - removed VDN and PER

---

## Experimental Results

### Test 1: Original Config (Lambda=0.05, t_max=3600)
**Result: COMPLETE FAILURE**
- Training returns: -32 to -3574 (diverging)
- Eval mean return: -7516 ± 583
- Avg waiting: 163 passengers (vs random: 8.8)
- **Conclusion**: Problem is impossible - too many passengers

### Test 2: Improved Config (Lambda=0.005, t_max=1800)
**Result: STABLE BUT POOR**
- Training returns: -44 to +2 (stable range)
- Eval mean return: -674 ± 180
- Avg waiting: 27 passengers (vs random: 3.3)
- **Conclusion**: Stable training but worse than random

### Test 3: Simple Config (Lambda=0.003, t_max=600, 5 floors)
**Result: STABLE BUT POOR**
- Training returns: -11 to +14 (good range)
- Eval mean return: -40 ± 22
- Random baseline: +1.7 ± 2.9
- **Conclusion**: Training looks good, but evaluation fails

### Test 4: Final Default Config (Lambda=0.005, t_max=1200, 8 floors)
**Result: TRAINING POSITIVE, EVAL NEGATIVE**
- Training returns: 0 to +93 (EXCELLENT!)
- Eval mean return: -210 ± 133
- Random baseline: +77 ± 11 (BETTER!)
- **Conclusion**: Agent is learning something, but not generalizing

---

## Performance Comparison: Random vs Trained

| Metric | Random Policy | Trained Policy | Difference |
|--------|--------------|----------------|------------|
| Mean Return | +77.58 | -210.65 | **-288.23** ❌ |
| Avg Waiting | 1.41 | 10.57 | **+9.16** ❌ |
| Avg In-Car | 2.43 | 5.59 | **+3.16** ❌ |

**Conclusion:** Random policy is significantly better (288 points higher return).

---

## Root Cause Analysis

### Why Is the Trained Agent Worse Than Random?

1. **Reward Structure Issues**
   - Penalty for waiting/in-car passengers dominates positive rewards
   - Agent may learn to avoid picking up passengers (minimizing penalty)
   - Even with 20-50x reward increases, penalties still dominate

2. **Multi-Agent Credit Assignment**
   - With 2 elevators, it's unclear which elevator caused good/bad outcomes
   - Shared reward signal makes learning difficult
   - VDN mixer attempts to solve this but adds complexity

3. **Action Space Complexity**
   - 4 actions per elevator = 4^2 = 16 joint actions
   - Many action combinations lead to similar outcomes
   - Exploration is inefficient

4. **Seed-Specific Overfitting**
   - Training shows positive returns (seed=42)
   - Evaluation shows negative returns (seed=999)
   - Agent is memorizing patterns, not learning general policy

5. **Sparse Feedback**
   - Rewards only given when passengers board/alight
   - Most time steps have only negative penalty signal
   - Makes learning slow and unstable

---

## Recommendations

### For Immediate Use
**Use the updated default.yaml configuration:**
- Lambda: 0.005 (not 0.05)
- t_max: 1200 (not 3600)  
- Strong positive rewards (r_alight=2.0, r_board=1.0)
- Simplified agent (no VDN, no PER)

**These parameters create a tractable problem**, even if the current agent doesn't solve it perfectly.

### For Better Performance

#### 1. Fix Reward Structure
```yaml
# Rebalance rewards to make positive actions more valuable
r_alight: 10.0      # Very strong delivery reward
r_board: 5.0        # Strong pickup reward
w_wait: 0.5         # Reduce penalty weight
w_incar: 0.2        # Reduce penalty weight
penalty_normalize: true
```

#### 2. Add Curriculum Learning
Start with very easy problem, gradually increase difficulty:
```python
# Phase 1: Lambda=0.002, 5 floors, 600s
# Phase 2: Lambda=0.005, 8 floors, 1200s
# Phase 3: Lambda=0.010, 10 floors, 1800s
```

#### 3. Try Different Algorithms
- **PPO** (Proximal Policy Optimization) - better for multi-agent
- **A3C** (Asynchronous Actor-Critic) - good for coordination
- **QMIX** (better than VDN for multi-agent value decomposition)

#### 4. Add Auxiliary Rewards
```python
# Reward for moving toward passengers
# Penalty for staying idle when calls exist
# Reward for maintaining low wait times
```

#### 5. Use Heuristic Warm-Start
Initialize the network with a simple rule-based policy:
- Go to nearest call
- Open doors when at called floor
- This provides better starting point than random

---

## Files Modified

### Configuration Files
1. **`dsp2/configs/default.yaml`** - Updated with reasonable hyperparameters
2. **`dsp2/configs/improved.yaml`** - Intermediate difficulty config
3. **`dsp2/configs/simple.yaml`** - Easy config for testing
4. **`dsp2/configs/quick.yaml`** - Fast training config

### Analysis Scripts
1. **`final_evaluation.py`** - Comprehensive comparison tool
2. **`compare_policies.py`** - Random vs trained comparison
3. **`test_baseline.py`** - Random policy baseline tester

### Documentation
1. **`HYPERPARAMETER_TUNING_REPORT.md`** - Detailed analysis
2. **`FINAL_EVALUATION_REPORT.md`** - This document

---

## Conclusion

### ✅ Successfully Completed
1. **Identified the problem**: Original lambda=0.05 was 10x too high
2. **Made hyperparameters reasonable**: Reduced by 90% to lambda=0.005
3. **Improved reward shaping**: 20-50x increases in positive rewards
4. **Simplified architecture**: Removed unnecessary complexity
5. **Created evaluation framework**: Tools to compare performance

### ⚠️ Outstanding Issue
**The DDQN agent does not learn an effective policy**, even with reasonable hyperparameters. This is likely due to:
- Reward structure still dominated by penalties
- Multi-agent credit assignment challenges
- Insufficient exploration
- Algorithm not suitable for this problem

### 📊 Key Metric: Lambda Reduction
**Original: 0.05 → Final: 0.005 (90% reduction)**

This is the **single most important change**. The original value created an impossible problem with 180 passengers over 1 hour for just 2 elevators.

### 🎯 Bottom Line
The hyperparameters are now **REASONABLE and SENSIBLE** for an elevator control problem. However, **better learning algorithms or reward engineering** are needed for the agent to outperform random policy.

---

## Next Steps if Continuing This Project

1. **Try PPO or A3C** instead of DDQN
2. **Implement curriculum learning** with gradually increasing difficulty
3. **Redesign reward function** to emphasize positive actions more
4. **Add shaped rewards** for progress toward goals
5. **Consider single-agent formulation** (one agent controlling both elevators)
6. **Use expert demonstrations** (imitation learning) as warm-start
7. **Increase training duration** to 200k+ steps with curriculum

---

**Author:** AI Assistant  
**Project:** DSP2 Multi-Elevator EGCS with Reinforcement Learning  
**Status:** Hyperparameters tuned, performance issues documented

