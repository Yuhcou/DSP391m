"""
Direct training and evaluation script with updated hyperparameters
"""
import os
import sys
import yaml
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dsp2.env.egcs_env import EGCSEnv
from dsp2.agents.ddqn_agent import DDQNAgent, AgentConfig
from dsp2.agents.replay import ReplayBuffer
from dsp2.agents.masks import legal_action_mask_from_state

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

def train():
    print("=" * 80)
    print("TRAINING WITH UPDATED HYPERPARAMETERS")
    print("=" * 80)
    print("\nKey Changes:")
    print("  - lambda: 0.05 -> 0.01 (more reasonable arrival rate)")
    print("  - lr: 0.0002 -> 0.0003 (slightly higher learning rate)")
    print("  - batch_size: 128 -> 64 (more stable updates)")
    print("  - replay_capacity: 200000 -> 100000 (more efficient)")
    print("  - target_update_steps: 10000 -> 5000 (faster target updates)")
    print("  - epsilon_end: 0.05 -> 0.01 (more exploitation)")
    print("  - decay_steps: 50000 -> 80000 (slower epsilon decay)")
    print("  - grad_clip: 5.0 -> 10.0 (allow larger gradients)")
    print("  - w_incar: 0.2 -> 0.3 (penalize in-car passengers more)")
    print("=" * 80)
    print()

    config_path = os.path.join('dsp2', 'configs', 'default.yaml')
    cfg = load_config(config_path)

    logdir = cfg['logdir']
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    env = make_env(cfg)

    agent_cfg = AgentConfig(
        gamma=cfg['gamma'],
        lr=cfg['lr'],
        batch_size=cfg['batch_size'],
        target_update_steps=cfg['target_update_steps'],
        tau=cfg['tau'],
        epsilon_start=cfg['epsilon_start'],
        epsilon_end=cfg['epsilon_end'],
        decay_steps=cfg['decay_steps'],
        grad_clip=cfg['grad_clip'],
        dueling=cfg['dueling'],
        use_per=cfg['use_per'],
        per_alpha=cfg['per_alpha'],
        per_beta=cfg['per_beta'],
        per_beta_increment=cfg['per_beta_increment'],
        use_vdn=cfg['use_vdn'],
        use_central_bias=cfg['use_central_bias']
    )

    agent = DDQNAgent(env.state_size, env.N, env.M, config=agent_cfg)
    buffer = ReplayBuffer(cfg['replay_capacity'], env.state_size, env.M, seed=cfg['seed'])

    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])

    s = env.reset()
    total_steps = cfg['training_steps']
    warmup = cfg['warmup_steps']
    ep_return = 0.0
    ep = 0

    def mask_fn(state_vec):
        return legal_action_mask_from_state(state_vec, env.N, env.M)

    print(f"Starting training for {total_steps} steps...")
    print(f"Environment: {env.N} floors, {env.M} elevators, capacity={env.capacity}")
    print(f"Lambda (arrival rate): {cfg['lambda']}")
    print()

    for t in range(total_steps):
        mask = env.action_mask()
        a = agent.select_action(s, mask, epsilon=None)
        s2, r, done, info = env.step(a)
        buffer.add(s, a, r, s2, done)
        s = s2
        ep_return += r

        if buffer.can_sample(agent_cfg.batch_size) and t >= warmup:
            batch = buffer.sample(agent_cfg.batch_size)
            loss, q_mean, _ = agent.train_step(batch, mask_fn)
            writer.add_scalar('loss', loss, t)
            writer.add_scalar('q_mean', q_mean, t)

        if done:
            writer.add_scalar('episode_return', ep_return, t)
            writer.add_scalar('n_waiting', info.get('n_waiting', 0), t)
            s = env.reset()
            ep_return = 0.0
            ep += 1

        if (t + 1) % cfg['log_interval'] == 0:
            print(f"Step {t+1}/{total_steps} | eps={agent.epsilon():.3f} | ep={ep} | waiting={info.get('n_waiting', 0)}")

        if (t + 1) % cfg['ckpt_interval'] == 0:
            ckpt_path = os.path.join(logdir, f'ckpt_step_{t+1}.pt')
            torch.save({'policy': agent.policy.state_dict(), 'cfg': agent_cfg.__dict__}, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # Save final checkpoint
    ckpt_final = os.path.join(logdir, 'ckpt_final.pt')
    torch.save({'policy': agent.policy.state_dict(), 'cfg': agent_cfg.__dict__}, ckpt_final)
    print(f"\nTraining complete! Final checkpoint saved: {ckpt_final}")

    writer.close()
    return env, agent, cfg

def evaluate(env, agent, cfg, n_episodes=10):
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)
    print(f"Running {n_episodes} evaluation episodes...\n")

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
            a = agent.select_action(s, mask, epsilon=0.0)  # Greedy
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

        print(f"Episode {ep+1}/{n_episodes}: Return={ep_ret:.2f}, Avg Waiting={avg_waiting:.2f}, Avg In-Car={avg_incar:.2f}")

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Mean Return:      {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
    print(f"Mean Avg Waiting: {np.mean(avg_waiting_list):.2f} ± {np.std(avg_waiting_list):.2f}")
    print(f"Mean Avg In-Car:  {np.mean(avg_incar_list):.2f} ± {np.std(avg_incar_list):.2f}")
    print("=" * 80)

    return {
        'episode_returns': episode_returns,
        'return_mean': np.mean(episode_returns),
        'return_std': np.std(episode_returns),
        'avg_waiting_mean': np.mean(avg_waiting_list),
        'avg_waiting_std': np.std(avg_waiting_list),
        'avg_incar_mean': np.mean(avg_incar_list),
        'avg_incar_std': np.std(avg_incar_list)
    }

if __name__ == "__main__":
    # Train
    env, agent, cfg = train()

    # Evaluate
    results = evaluate(env, agent, cfg, n_episodes=10)

    print("\nDone! Check dsp2/logs/ for TensorBoard logs.")

