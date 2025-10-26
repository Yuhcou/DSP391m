# training/train_dqn.py
import os
import numpy as np
from agents.dqn_agent import DQNAgent

def train_agent(env, episodes=200, save_path="results/models/dqn_elevator.pth"):
    state_dim = env.num_elevators * 2
    action_dim = env.num_elevators

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    eps = 1.0
    eps_min = 0.05
    eps_decay = 0.995

    all_rewards = []
    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            # agent action
            action = agent.select_action(state, epsilon=eps)
            # step environment
            next_state, reward, done, info = env.step(action)
            # store transition
            agent.remember(state, action, reward, next_state, float(done))
            # train
            loss = agent.update()
            state = next_state
            ep_reward += reward
            steps += 1

        # decay epsilon
        eps = max(eps_min, eps * eps_decay)
        agent.epsilon = eps  # keep agent epsilon consistent
        all_rewards.append(ep_reward)

        if ep % 10 == 0 or ep == 1:
            avg_recent = np.mean(all_rewards[-10:])
            print(f"[Ep {ep}/{episodes}] reward={ep_reward:.2f} avg10={avg_recent:.2f} eps={eps:.3f}")

        # save periodically
        if ep % 50 == 0:
            agent.save(save_path)

    # final save
    agent.save(save_path)
    print(f"Training complete. Model saved to {save_path}")
    return save_path
