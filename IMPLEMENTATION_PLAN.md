# Implementation Plan — EGCS (Multi‑Elevator) + DDQN

Goal: implement a Gym‑style multi‑elevator simulator and a DDQN agent (M elevators × 4 actions each). Deliver working training pipeline, evaluation scripts and reproducible experiments.

1. Repo layout (suggested)
- dsp2/
  - env/
    - __init__.py
    - egcs_env.py           # Gym-like Env class (multi-elevator)
    - sim_helpers.py        # passenger queues, arrival/destination utils
  - agents/
    - __init__.py
    - ddqn_agent.py         # agent class: networks, select_action(), train_step()
    - replay.py             # replay buffer (uniform / prioritized optional)
    - networks.py           # NN models (policy + target)
    - masks.py              # action masking utilities
  - train/
    - train.py              # training loop + checkpoints
    - eval.py               # evaluation script (multi‑seed)
  - tests/
    - test_env.py
    - test_agent.py
  - logs/                   # tensorboard / checkpoints
  - configs/
    - default.yaml
  - IMPLEMENTATION_PLAN.md  # this file
  - README.md

2. Environment: egcs_env.py (Gym-like)
- Class: EGCSEnv(gym.Env)
  - __init__(N_floors, M_elevators, capacity, dt, lambda_fn, T_max, seed)
  - reset() -> observation (numpy array shaped State_size)
  - step(action_vector) -> (obs, reward, done, info)
  - render() optional
- State vector construction:
  - positions: int array (M,)
  - directions: int array (M,) values −1/0/1
  - hall_up, hall_down: binary arrays (N,)
  - car_calls: binary matrix (M,N) flattened
  - concat in fixed order; dtype float32
- Action format: array length M, each in {0,1,2,3}
- Step internals (per timestep):
  1. Validate/mask each elevator action (top/bottom floor, moving/door rules)
  2. Apply move/stop/open effects to per‑elevator state
  3. For elevators with Open: alight -> remove onboard with dest==floor; board from matching queue up to capacity; update car_calls/hall calls
  4. Generate arrivals: Bernoulli p_t = 1 − exp(−λ(t)·Δt), sample s,d, add to floor queue
  5. Compute scalar reward: e.g., r = −(w_wait * n_waiting + w_incar * n_incar)
  6. Return next state (vectorized)
- Deterministic masking function: returns boolean mask shape (M,4) marking legal actions

3. Agent design (agents/ddqn_agent.py & networks.py)
- Network:
  - Input dim = State_size = M*N + 2N + 2M
  - Shared backbone (e.g., FC 128→128 ReLU) → Linear output M*4
  - Reshape output to (M,4)
- Action selection:
  - Q = policy_net(state) → (M,4)
  - mask illegal actions per elevator (set to −1e9)
  - per‑elevator ε‑greedy (flip coin per elevator) or joint as config
  - return action vector
- Replay buffer:
  - store transitions: (s, a_vector, r, s', done)
  - when sampling, return arrays: s: (B, S), a: (B, M), r: (B,), s': (B, S), done: (B,)
- Training step (vectorized DDQN):
  1. Q_next_policy = policy_net(s') -> (B, M, 4)
  2. Mask illegal actions in s' to get Q_masked_policy
  3. a_star = argmax_a Q_masked_policy -> (B, M) (indices per elevator)
  4. Q_target = target_net(s') -> (B, M, 4)
  5. Gather Q_target_eval = Q_target[range(B), :, a_star] -> (B, M)
  6. Y = r[:,None] + γ * Q_target_eval * (1 - done[:,None])
  7. Q_current = policy_net(s) -> (B, M, 4); pick Q_current_taken = Q_current[range(B), :, a_taken] -> (B,M)
  8. Loss = MSE(Q_current_taken, Y) averaged over B and M
  9. Backprop and update policy; periodic target update (hard) or Polyak (soft)
- Implementation notes:
  - Use vectorized gather ops in PyTorch/TensorFlow
  - Ensure masking used both for selection (a_star) and for action choice
  - Reward is scalar shared across elevators; targets computed per-elevator as above

4. Training loop (train/train.py)
- Seed rngs (python, numpy, torch)
- Instantiate env and agent
- Warmup: collect random transitions until buffer >= batch_size
- Main loop:
  - select action via agent.select_action(state, epsilon)
  - step env -> next_state, r, done
  - store transition
  - agent.train_step() every step (or every K steps)
  - handle episode end: reset env, log metrics
  - periodically save checkpoints and evaluate
- Logging: use Tensorboard for:
  - episode reward, avg waiting time, avg journey time, hall calls cleared, loss, epsilon, learning rate
- Evaluation: deterministic policy (epsilon=0) over multiple seeds and traffic scenarios

5. Tests
- test_env.py:
  - single-step transitions, action masking correctness, passenger boarding/alighting invariants, state size
- test_agent.py:
  - network input/output shapes for sample state, gather ops correctness, mask consistency
- unit tests for replay buffer (sample shapes) and target computation (small B, M hand-verified)

6. Hyperparameters (start)
- gamma = 0.99
- lr = 1e-4 (Adam)
- batch_size = 64
- replay_capacity = 200_000
- target_update_steps = 10_000 (or tau=1e-3 for Polyak)
- epsilon_start=1.0, epsilon_end=0.05, decay_steps=200_000
- network: FC(128)->FC(128)
- training_steps: dependent on compute (e.g., 1e6 env steps)

7. Practical tips & improvements
- Use centralized training + decentralized execution (CTDE) or VDN/QMIX if coordination required
- Consider prioritized replay and dueling networks to speed convergence
- Use gradient clipping and learning rate schedules
- Curriculum: progressively increase traffic intensity and variance
- For scalability: vectorize env for parallel actors (multiple env instances) for sample throughput

8. Deliverables & checkpoints
- Week 1: env/egcs_env.py basic multi-elevator step/reset and tests
- Week 2: agent skeleton (networks, replay, masks) + unit tests
- Week 3: training loop, logging, first small experiment (converges on simple traffic)
- Week 4: larger experiments, evaluation scripts, hyperparameter sweep, writeup

9. Reproducibility
- Save: git commit hash, config YAML, random seeds, and checkpointed networks
- Store logs and evaluation seeds under logs/ for result comparison

# Implementation status

- Steps 1–6 scaffolded:
  - Package `dsp2/` with env, agents, train, tests, configs.
  - Basic EGCSEnv with state, action mask, step, arrivals, reward.
  - DDQN agent, networks, replay, masking utilities.
  - Training and evaluation scripts.
  - Unit tests for env and agent shapes.
  - Default config and README.

Notes (implementation pitfalls)
- Keep action masking consistent between selection and learning (both policy and target when selecting a*)
- Watch batch shapes carefully when gathering per‑elevator actions
- Reward shaping: use simple scalar first; experiment with small bonuses for clearing hall calls
- Test env determinism with fixed seed and fixed arrival schedule for unit tests

End.
