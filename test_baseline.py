"""
Test random policy vs trained policy to understand baseline performance
"""
import os
import sys
import yaml
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dsp2.env.egcs_env import EGCSEnv

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def make_env(cfg):
    return EGCSEnv(
        n_floors=cfg['n_floors'],
        m_elevators=cfg['m_elevators'],
        capacity=cfg['capacity'],
        dt=cfg['dt'],
        lambda_fn=lambda t: cfg['lambda'],
        t_max=cfg['t_max'],
        seed=cfg['seed'],
        w_wait=cfg['w_wait'],
        w_incar=cfg['w_incar'],
        r_alight=cfg['r_alight'],
        r_board=cfg['r_board'],
        penalty_normalize=cfg['penalty_normalize']
    )

def random_policy_eval(env, n_episodes=5):
    """Evaluate random policy as baseline"""
    print("=" * 80)
    print("RANDOM POLICY BASELINE")
    print("=" * 80)

    episode_returns = []
    avg_waiting_list = []
    avg_incar_list = []

    for ep in range(n_episodes):
        s = env.reset()
        done = False
        ep_ret = 0.0
        steps = 0
        waiting_acc = 0.0
        incar_acc = 0.0

        while not done:
            mask = env.action_mask()
            # Random action among legal actions for each elevator
            a = np.zeros(env.M, dtype=int)
            for elev_idx in range(env.M):
                legal = [act for act in range(4) if mask[elev_idx, act]]
                if legal:
                    a[elev_idx] = np.random.choice(legal)
                else:
                    a[elev_idx] = 0  # default to stay

            s, r, done, info = env.step(a)
            ep_ret += r
            steps += 1
            waiting_acc += info.get('n_waiting', 0)
            incar_acc += info.get('n_incar', 0)

        avg_waiting = waiting_acc / max(1, steps)
        avg_incar = incar_acc / max(1, steps)

        episode_returns.append(ep_ret)
        avg_waiting_list.append(avg_waiting)
        avg_incar_list.append(avg_incar)

        print(f"Episode {ep+1}: Return={ep_ret:.2f}, Avg Waiting={avg_waiting:.2f}, Avg In-Car={avg_incar:.2f}")

    print("\nRandom Policy Results:")
    print(f"Mean Return:      {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
    print(f"Mean Avg Waiting: {np.mean(avg_waiting_list):.2f} ± {np.std(avg_waiting_list):.2f}")
    print(f"Mean Avg In-Car:  {np.mean(avg_incar_list):.2f} ± {np.std(avg_incar_list):.2f}")
    print("=" * 80)

if __name__ == "__main__":
    config_path = os.path.join('dsp2', 'configs', 'quick.yaml')
    cfg = load_config(config_path)
    env = make_env(cfg)

    print(f"Environment Config:")
    print(f"  Lambda (arrival rate): {cfg['lambda']}")
    print(f"  Floors: {cfg['n_floors']}, Elevators: {cfg['m_elevators']}")
    print(f"  Capacity: {cfg['capacity']}, Max Time: {cfg['t_max']}")
    print(f"  Weights: w_wait={cfg['w_wait']}, w_incar={cfg['w_incar']}")
    print()

    random_policy_eval(env, n_episodes=5)

