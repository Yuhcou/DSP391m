# main.py
from envs.elevator_env import ElevatorEnv
from training.train_dqn import train_agent
from training.evaluate import evaluate_agent
from agents.classical_dispatcher import ClassicalDispatcher
import os

if __name__ == "__main__":
    os.makedirs("results/models", exist_ok=True)
    # env configuration
    env = ElevatorEnv(num_elevators=6, num_floors=15, episode_length=200)

    # Train RL agent
    model_path = train_agent(env, episodes=2000, save_path="results/models/dqn_elevator.pth")

    # Evaluate RL agent vs classical baseline
    baseline = ClassicalDispatcher(num_elevators=6, num_floors=15, mode='nearest_car')
    evaluate_agent(env, model_path, baseline, episodes=50)
