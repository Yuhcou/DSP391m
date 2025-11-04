# REINFORCEMENT LEARNING PROGRAMMING PROJECT

Report — Simulation Method & RL Agent (combined)

– Hanoi, October 2025 –

...existing code...

## I. Overview

- Purpose: provide a single, consistent specification for the Gym‑style EGCS environment (multi‑elevator) and the RL agent (DDQN) used to control it.
- Short diagnosis of prior inconsistency: Report 2 described a single elevator; Report 3 assumed M elevators. This combined document makes M explicit and fixes a malformed arrival formula.

...existing code...

## II. Simulation Method (Environment)

Parameters
- N — number of floors
- M — number of elevators
- Δt — simulation timestep
- T_max — episode duration
- λ(t) — time‑varying arrival rate
- capacity_e — per‑elevator capacity (optional)

### 1. State (Observation) Space

For N floors and M elevators the state S_t is a fixed vector concatenating:

- Elevator positions: length M (integer floor indices 0..N−1)  
- Elevator directions: length M (−1 down, 0 idle, +1 up)  
- Hall calls up: length N (binary)  
- Hall calls down: length N (binary)  
- Per‑car calls: M blocks each length N (binary destinations selected by onboard passengers)

Total input size:
State_size = M*N (car calls) + 2N (hall calls) + 2M (positions + directions)

### 2. Action Space

At each timestep the agent issues a vector action A_t = [A_t^e]_{e=1..M}, one high‑level command per elevator:
Ae ∈ {0: Move Up, 1: Move Down, 2: Stop/Idle, 3: Open Door}

To avoid 4^M joint explosion we implement the policy to emit M separate 4‑way Q blocks and select per‑elevator actions (optionally coordinated via shared network and joint experience).

### 3. Reward

Same scalar reward as before: negative per waiting passenger and per in‑elevator passenger per timestep (can be weighted). Reward is global (shared) and used uniformly for computing per‑elevator TD targets.

### 4. Arrival / Destination Model

- Arrivals: non‑homogeneous Poisson rate λ(t). For discrete Δt the probability of at least one arrival in Δt is:
  p_t = 1 − exp(−λ(t) · Δt)
- When an arrival occurs sample start floor s and destination d from P(d | s, t) (time‑of‑day conditional) as described previously (morning up‑peak, midday interfloor, evening down‑peak).

### 5. Simulation Dynamics (step)

Procedure (per timestep t):
1. Read agent action vector A_t = [A_t^e] (one per elevator).
2. For each elevator e = 1..M:
   - Mask illegal actions (e.g., Move Up at top floor) — these are not executed by simulator.
   - Execute permitted action: update position/direction/door state. If Open Door then elevator enters boarding/alighting phase at its current floor.
3. For elevators that opened doors: perform alighting (remove onboard passengers whose destination matches floor), then boarding (queue up matching direction or all if idle) up to capacity_e; update per‑car calls and hall calls.
4. Generate new passengers: with probability p_t = 1 − exp(−λ(t)·Δt) create arrival(s) and place them in floor queues; set hall calls accordingly.
5. Compute scalar reward R_{t+1} (sum over waiting + in‑car passengers).
6. Assemble next state S_{t+1} (concatenated M positions, M directions, 2N hall calls, M×N car calls).
7. done ← (t ≥ T_max)

Pseudocode (high‑level) — the order preserves original algorithm but applies per elevator where needed.

...existing code...

## III. Reinforcement Learning Agent (DDQN) — multi‑elevator

### 1. High level

- Algorithm: Double Deep Q‑Network (DDQN)
- State input: State_size (see above)
- Network output: M × 4 Q‑values (concatenated blocks, one 4‑way head per elevator)
- Reward: global scalar

Rationale: minimal change from single‑elevator DDQN — we vectorize outputs/targets per elevator and keep the DDQN decoupling (policy net selects a* per elevator, target net evaluates).

### 2. Network and Action Selection

- Input dimension = State_size = M*N + 2N + 2M
- Output dimension = M * 4, shaped (M,4)
- For a given s:
  1. Q = Q_theta(s) → shape (M,4)
  2. For each elevator e: mask illegal actions (set to −∞)
  3. Per‑elevator ε‑greedy: with probability ε_e pick a random feasible action for elevator e; otherwise pick argmax_a Q_masked[e,a]
  4. Form A_t = [a_e]_{e=1..M} and step environment.

Note: exploration can be per‑elevator (independent coins) or joint; per‑elevator is simpler and often effective.

### 3. Replay and DDQN target (vectorized)

- Stored transitions: (s, a_vector, r, s', done)
- During training sample a minibatch B.
- For each sample j and each elevator e:
  - Q_next_policy = Q_theta(s'j)  # shape (M,4)
  - Mask illegal actions in s'j per elevator
  - a*_e = argmax_a Q_next_policy[e,a]   (policy net selects)
  - If done_j: Y_{j,e} = r_{j+1}
    Else: Y_{j,e} = r_{j+1} + γ * Q_theta'(s'j)[e, a*_e]  (target net evaluates)
- Loss_j = mean_e ( Q_theta(s_j)[e, a_{j,e}] − Y_{j,e} )^2
- Final loss = mean_j Loss_j  (average over batch and elevators)

This preserves DDQN decoupling while producing per‑elevator targets and a single scalar loss.

### 4. Training loop (concise)

- Initialize Q_theta, Q_theta' ← Q_theta, Replay buffer D
- For steps/episodes:
  - Select A_t per‑elevator using masked ε‑greedy on Q_theta
  - Step env: receive r, s'
  - Store (s, A_t, r, s', done)
  - If buffer large enough sample batch and compute vectorized targets/loss as above; gradient step on θ
  - Update θ' periodically (hard) or via Polyak averaging (recommended)
  - Decay ε

### 5. Action masking

- Apply same mask during selection and when computing a*_e for target calculation.
- Mask illegal actions both at execution and in learning to avoid spurious Q updates.

## IV. Fixes and Notes

- Fixed arrival probability: p_t = 1 − exp(−λ(t)·Δt) (was malformed).  
- Unified state/action description for M elevators so Reports 2 and 3 are consistent.  
- Kept DDQN algorithm intact; changes are vectorization of outputs/targets and per‑elevator masking.

## V. Practical improvements (brief)

- Centralized training with decentralized execution (CTDE), or multi‑agent value decomposition (VDN/QMIX) for coordination.  
- Factorised heads + parameter sharing reduce sample complexity.  
- Prioritized Experience Replay, distributional RL, dueling heads, and soft (Polyak) target updates for stability.  
- Curriculum traffic generation and small reward shaping (e.g., bonus for clearing hall calls) to speed convergence.

## VI. References

...[existing references: Mnih et al., Van Hasselt et al., Gymnasium, Pawletta & Bartelt]...


