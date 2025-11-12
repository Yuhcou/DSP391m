from __future__ import annotations
import os
import time
from dataclasses import asdict
import argparse
import yaml
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from dsp2.env.egcs_env import EGCSEnv
from dsp2.agents.ddqn_agent import DDQNAgent, AgentConfig
from dsp2.agents.replay import ReplayBuffer
from dsp2.agents.masks import legal_action_mask_from_state


def load_yaml_config(path: str | None) -> dict:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping/object, got {type(data).__name__}")
    return data

def make_env(cfg: dict) -> EGCSEnv:
    env = EGCSEnv(
        n_floors=cfg.get('n_floors', 10),
        m_elevators=cfg.get('m_elevators', 2),
        capacity=cfg.get('capacity', 8),
        dt=cfg.get('dt', 1.0),
        lambda_fn=lambda t: cfg.get('lambda', 0.05),
        t_max=cfg.get('t_max', 3600),
        seed=cfg.get('seed', 42),
        w_wait=cfg.get('w_wait', 1.0),
        w_incar=cfg.get('w_incar', 0.2),
        r_alight=cfg.get('r_alight', 0.1),
        r_board=cfg.get('r_board', 0.02),
        penalty_normalize=cfg.get('penalty_normalize', True),
    )
    return env


def main(config_path: str):
    cfg = load_yaml_config(config_path)
    logdir = cfg.get('logdir', os.path.join('dsp2', 'logs'))
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    env = make_env(cfg)
    # Adjust defaults per recommendations if not explicitly set in config
    decay_steps = cfg.get('decay_steps', 50_000 if cfg.get('training_steps', 100_000) >= 100_000 else 20_000)
    batch_size = cfg.get('batch_size', 128 if torch.cuda.is_available() else 64)
    lr = cfg.get('lr', 2e-4 if torch.cuda.is_available() else 1e-4)

    agent_cfg = AgentConfig(
        gamma=cfg.get('gamma', 0.99),
        lr=lr,
        batch_size=batch_size,
        target_update_steps=cfg.get('target_update_steps', 10000),
        tau=cfg.get('tau', 1.0),
        epsilon_start=cfg.get('epsilon_start', 1.0),
        epsilon_end=cfg.get('epsilon_end', 0.05),
        decay_steps=decay_steps,
        grad_clip=cfg.get('grad_clip', 5.0),
        dueling=cfg.get('dueling', True),
        use_per=cfg.get('use_per', False),
        per_alpha=cfg.get('per_alpha', 0.6),
        per_beta=cfg.get('per_beta', 0.4),
        per_beta_increment=cfg.get('per_beta_increment', 0.0),
        use_vdn=cfg.get('use_vdn', True),
        use_central_bias=cfg.get('use_central_bias', False),
    )
    agent = DDQNAgent(env.state_size, env.N, env.M, config=agent_cfg)

    use_per = bool(cfg.get('use_per', False))
    if use_per:
        try:
            from dsp2.agents.replay import PERBuffer  # lazy import
            buffer = PERBuffer(cfg.get('replay_capacity', 200_000), env.state_size, env.M,
                               alpha=agent_cfg.per_alpha, beta=agent_cfg.per_beta,
                               beta_increment=agent_cfg.per_beta_increment,
                               seed=cfg.get('seed', 42))
        except Exception as e:
            print(f"[WARN] PER unavailable ({e}); falling back to uniform ReplayBuffer.")
            buffer = ReplayBuffer(cfg.get('replay_capacity', 200_000), env.state_size, env.M, seed=cfg.get('seed', 42))
            use_per = False
    else:
        buffer = ReplayBuffer(cfg.get('replay_capacity', 200_000), env.state_size, env.M, seed=cfg.get('seed', 42))

    np.random.seed(cfg.get('seed', 42))
    torch.manual_seed(cfg.get('seed', 42))

    s = env.reset()
    total_steps = cfg.get('training_steps', 100_000)
    warmup = cfg.get('warmup_steps', agent_cfg.batch_size)
    ep_return = 0.0
    ep = 0

    def mask_fn(state_vec: np.ndarray):
        return legal_action_mask_from_state(state_vec, env.N, env.M)

    for t in range(total_steps):
        mask = env.action_mask()
        a = agent.select_action(s, mask, epsilon=None)
        s2, r, done, info = env.step(a)
        buffer.add(s, a, r, s2, done)
        s = s2
        ep_return += r

        if buffer.can_sample(agent_cfg.batch_size) and t >= warmup:
            if use_per:
                (batch, idx, w) = buffer.sample(agent_cfg.batch_size)
                loss, q_mean, per_td = agent.train_step(batch, mask_fn, per_indices=idx, per_weights=w)
                buffer.update_priorities(idx, per_td)
            else:
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

        if (t + 1) % cfg.get('log_interval', 1000) == 0:
            print(f"Step {t+1}/{total_steps} eps={agent.epsilon():.3f} return={ep_return:.1f} waiting={info.get('n_waiting',0)}")

        if (t + 1) % cfg.get('ckpt_interval', 50_000) == 0:
            ckpt_path = os.path.join(logdir, f'ckpt_step_{t+1}.pt')
            torch.save({'policy': agent.policy.state_dict(), 'cfg': asdict(agent_cfg)}, ckpt_path)

    # Save final checkpoint for evaluation
    ckpt_final = os.path.join(logdir, 'ckpt_final.pt')
    torch.save({'policy': agent.policy.state_dict(), 'cfg': asdict(agent_cfg)}, ckpt_final)

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=os.path.join('dsp2', 'configs', 'default.yaml'))
    args = parser.parse_args()
    main(args.config)
