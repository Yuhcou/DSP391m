# training/evaluate.py
import numpy as np
from agents.dqn_agent import DQNAgent
from utils.metrics import compute_basic_metrics
import torch

def evaluate_agent(env, model_path, baseline, episodes=50):
    state_dim = env.num_elevators * 2
    action_dim = env.num_elevators

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    agent.load(model_path)

    rl_rewards = []
    base_rewards = []

    # RL evaluation
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_r = 0.0
        while not done:
            action = agent.select_action(state, epsilon=0.0)
            next_state, r, done, info = env.step(action)
            state = next_state
            total_r += r
        rl_rewards.append(total_r)

    # Baseline evaluation (nearest_car)
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_r = 0.0
        while not done:
            state_dict = env.get_state_dict()
            # use current call floor from env (we don't have it outside step),
            # emulate by sampling a hall call here and letting baseline act on it,
            # then pass that chosen elevator to env.step()
            # For fair comparison, call env.step with the baseline's action (environment generates new call inside step).
            # So we just query baseline for a candidate elevator using a sampled floor:
            hall_floor = np.random.randint(0, env.num_floors)
            chosen = baseline.select_action(state_dict, hall_floor)
            next_state, r, done, info = env.step(chosen)
            state = next_state
            total_r += r
        base_rewards.append(total_r)

    print(f"\nEvaluation over {episodes} episodes:")
    print(f"Average RL total reward: {np.mean(rl_rewards):.3f} (std {np.std(rl_rewards):.3f})")
    print(f"Average Baseline total reward: {np.mean(base_rewards):.3f} (std {np.std(base_rewards):.3f})")

    # basic metrics for RL run (on last evaluation episode)
    metrics = compute_basic_metrics(rl_rewards)
    print("Sample metrics (RL rewards):", metrics)
