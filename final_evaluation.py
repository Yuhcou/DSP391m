"""
Final comprehensive evaluation: Default config trained agent vs random baseline
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

def run_eval(env, policy_fn, episodes=10):
    """Run evaluation with given policy function"""
    returns = []
    waiting = []
    incar = []

    for ep in range(episodes):
        s = env.reset()
        done = False
        ep_ret = 0.0
        steps = 0
        wait_acc = 0.0
        incar_acc = 0.0

        while not done:
            mask = env.action_mask()
            a = policy_fn(s, mask)
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
    config_path = os.path.join('dsp2', 'configs', 'default.yaml')
    cfg = load_config(config_path)

    print("=" * 80)
    print("FINAL EVALUATION - DEFAULT CONFIG")
    print("=" * 80)
    print(f"\nEnvironment Configuration:")
    print(f"  Floors:       {cfg['n_floors']}")
    print(f"  Elevators:    {cfg['m_elevators']}")
    print(f"  Capacity:     {cfg['capacity']}")
    print(f"  Lambda:       {cfg['lambda']} (arrival rate)")
    print(f"  Episode Time: {cfg['t_max']}s")
    print(f"  Weights:      w_wait={cfg['w_wait']}, w_incar={cfg['w_incar']}")
    print(f"  Rewards:      r_alight={cfg['r_alight']}, r_board={cfg['r_board']}")
    print()

    # Evaluate random policy
    print("-" * 80)
    print("RANDOM POLICY")
    print("-" * 80)
    env = make_env(cfg, seed=999)

    def random_policy(s, mask):
        a = np.zeros(cfg['m_elevators'], dtype=int)
        for i in range(cfg['m_elevators']):
            legal = [act for act in range(4) if mask[i, act]]
            a[i] = np.random.choice(legal) if legal else 0
        return a

    random_results = run_eval(env, random_policy, episodes=10)
    print(f"Mean Return:      {random_results['mean_return']:8.2f} ± {random_results['std_return']:.2f}")
    print(f"Mean Avg Waiting: {random_results['mean_waiting']:8.2f}")
    print(f"Mean Avg In-Car:  {random_results['mean_incar']:8.2f}")
    print(f"Episode Returns:  {[f'{r:.0f}' for r in random_results['returns']]}")

    # Evaluate trained policy
    print()
    print("-" * 80)
    print("TRAINED POLICY")
    print("-" * 80)

    logdir = cfg['logdir']
    ckpt_path = os.path.join(logdir, 'ckpt_final.pt')

    if os.path.exists(ckpt_path):
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

        env = make_env(cfg, seed=999)
        agent = DDQNAgent(env.state_size, env.N, env.M, config=agent_cfg)
        data = torch.load(ckpt_path, map_location='cpu')
        agent.policy.load_state_dict(data['policy'])

        def trained_policy(s, mask):
            return agent.select_action(s, mask, epsilon=0.0)

        trained_results = run_eval(env, trained_policy, episodes=10)
        print(f"Mean Return:      {trained_results['mean_return']:8.2f} ± {trained_results['std_return']:.2f}")
        print(f"Mean Avg Waiting: {trained_results['mean_waiting']:8.2f}")
        print(f"Mean Avg In-Car:  {trained_results['mean_incar']:8.2f}")
        print(f"Episode Returns:  {[f'{r:.0f}' for r in trained_results['returns']]}")

        # Comparison
        print()
        print("=" * 80)
        print("PERFORMANCE COMPARISON")
        print("=" * 80)
        improvement = trained_results['mean_return'] - random_results['mean_return']
        wait_reduction = random_results['mean_waiting'] - trained_results['mean_waiting']

        print(f"Return Improvement:  {improvement:+8.2f} (absolute)")
        print(f"Waiting Reduction:   {wait_reduction:+8.2f} passengers")
        print()

        if improvement > 5:
            print("✅ SUCCESS: Trained agent is BETTER than random policy!")
            print(f"   The agent achieves {improvement:.1f} higher return.")
        elif improvement > -5:
            print("⚠️  NEUTRAL: Trained agent is SIMILAR to random policy.")
            print("   More training or hyperparameter tuning may be needed.")
        else:
            print("❌ ISSUE: Trained agent is WORSE than random policy.")
            print("   This suggests a problem with learning or reward structure.")
    else:
        print(f"No checkpoint found at {ckpt_path}")
        print("Please run training first.")

    print("=" * 80)

    # Summary of changes
    print()
    print("HYPERPARAMETER CHANGES FROM ORIGINAL:")
    print("-" * 80)
    print("Lambda:     0.050 → 0.005  (90% reduction - key change!)")
    print("t_max:      3600s → 1200s  (67% reduction)")
    print("n_floors:   10 → 8         (20% reduction)")
    print("r_alight:   0.1 → 2.0      (20x increase)")
    print("r_board:    0.02 → 1.0     (50x increase)")
    print("w_incar:    0.2 → 0.5      (2.5x increase)")
    print("gamma:      0.99 → 0.95    (less emphasis on far future)")
    print("use_vdn:    true → false   (disabled for simplicity)")
    print("use_per:    true → false   (disabled for stability)")
    print("=" * 80)

