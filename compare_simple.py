"""
Compare simple config trained agent vs random baseline
"""
import os
import sys
import yaml
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dsp2.env.egcs_env import EGCSEnv
from dsp2.agents.ddqn_agent import DDQNAgent, AgentConfig

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def make_env(cfg, seed=None):
    return EGCSEnv(
        n_floors=cfg['n_floors'],
        m_elevators=cfg['m_elevators'],
        capacity=cfg['capacity'],
        dt=cfg['dt'],
        lambda_fn=lambda t: cfg['lambda'],
        t_max=cfg['t_max'],
        seed=seed if seed is not None else cfg['seed'],
        w_wait=cfg['w_wait'],
        w_incar=cfg['w_incar'],
        r_alight=cfg['r_alight'],
        r_board=cfg['r_board'],
        penalty_normalize=cfg['penalty_normalize']
    )

def random_eval(cfg, n_episodes=10):
    returns = []
    waiting = []
    incar = []

    for ep in range(n_episodes):
        env = make_env(cfg, seed=100+ep)
        s = env.reset()
        done = False
        ep_ret = 0.0
        steps = 0
        wait_acc = 0.0
        incar_acc = 0.0

        while not done:
            mask = env.action_mask()
            a = np.zeros(env.M, dtype=int)
            for i in range(env.M):
                legal = [act for act in range(4) if mask[i, act]]
                a[i] = np.random.choice(legal) if legal else 0

            s, r, done, info = env.step(a)
            ep_ret += r
            steps += 1
            wait_acc += info.get('n_waiting', 0)
            incar_acc += info.get('n_incar', 0)

        returns.append(ep_ret)
        waiting.append(wait_acc / max(1, steps))
        incar.append(incar_acc / max(1, steps))

    return {
        'returns': returns,
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'mean_waiting': np.mean(waiting),
        'mean_incar': np.mean(incar)
    }

def trained_eval(cfg, config_path, n_episodes=10):
    logdir = cfg['logdir']
    ckpt_path = os.path.join(logdir, 'ckpt_final.pt')

    if not os.path.exists(ckpt_path):
        print(f"No checkpoint found at {ckpt_path}")
        return None

    agent_cfg = AgentConfig(
        gamma=cfg['gamma'],
        lr=cfg['lr'],
        batch_size=cfg['batch_size'],
        target_update_steps=cfg['target_update_steps'],
        tau=cfg['tau'],
        epsilon_start=0.0,
        epsilon_end=0.0,
        decay_steps=1,
        grad_clip=cfg['grad_clip'],
        dueling=cfg['dueling'],
        use_vdn=cfg['use_vdn'],
        use_central_bias=cfg.get('use_central_bias', False)
    )

    returns = []
    waiting = []
    incar = []

    for ep in range(n_episodes):
        env = make_env(cfg, seed=100+ep)
        agent = DDQNAgent(env.state_size, env.N, env.M, config=agent_cfg)
        data = torch.load(ckpt_path, map_location='cpu')
        agent.policy.load_state_dict(data['policy'])

        s = env.reset()
        done = False
        ep_ret = 0.0
        steps = 0
        wait_acc = 0.0
        incar_acc = 0.0

        while not done:
            mask = env.action_mask()
            a = agent.select_action(s, mask, epsilon=0.0)
            s, r, done, info = env.step(a)
            ep_ret += r
            steps += 1
            wait_acc += info.get('n_waiting', 0)
            incar_acc += info.get('n_incar', 0)

        returns.append(ep_ret)
        waiting.append(wait_acc / max(1, steps))
        incar.append(incar_acc / max(1, steps))

    return {
        'returns': returns,
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'mean_waiting': np.mean(waiting),
        'mean_incar': np.mean(incar)
    }

if __name__ == "__main__":
    config_path = os.path.join('dsp2', 'configs', 'simple.yaml')
    cfg = load_config(config_path)

    print("=" * 80)
    print("AGENT COMPARISON - SIMPLE CONFIG")
    print("=" * 80)
    print(f"\nEnvironment: lambda={cfg['lambda']}, t_max={cfg['t_max']}s")
    print(f"             {cfg['n_floors']} floors, {cfg['m_elevators']} elevators, capacity={cfg['capacity']}\n")

    print("Evaluating RANDOM policy...")
    random_results = random_eval(cfg, n_episodes=10)
    print(f"  Return:  {random_results['mean_return']:.2f} ± {random_results['std_return']:.2f}")
    print(f"  Waiting: {random_results['mean_waiting']:.2f}")
    print(f"  In-Car:  {random_results['mean_incar']:.2f}")
    print(f"  Returns: {[f'{r:.1f}' for r in random_results['returns']]}")

    print("\nEvaluating TRAINED policy...")
    trained_results = trained_eval(cfg, config_path, n_episodes=10)
    if trained_results:
        print(f"  Return:  {trained_results['mean_return']:.2f} ± {trained_results['std_return']:.2f}")
        print(f"  Waiting: {trained_results['mean_waiting']:.2f}")
        print(f"  In-Car:  {trained_results['mean_incar']:.2f}")
        print(f"  Returns: {[f'{r:.1f}' for r in trained_results['returns']]}")

        print("\n" + "=" * 80)
        print("COMPARISON")
        print("=" * 80)
        improvement = trained_results['mean_return'] - random_results['mean_return']
        print(f"Return Improvement:  {improvement:+.2f} (absolute)")

        wait_reduction = random_results['mean_waiting'] - trained_results['mean_waiting']
        print(f"Waiting Reduction:   {wait_reduction:+.2f} passengers (absolute)")

        if improvement > 1.0:
            print("\n✅ Agent is BETTER than random policy!")
        elif improvement > -1.0:
            print("\n⚠️  Agent is SIMILAR to random policy")
        else:
            print("\n❌ Agent is WORSE than random policy")
    else:
        print("  Could not load trained model")

    print("=" * 80)

