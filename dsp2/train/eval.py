from __future__ import annotations
import os
import argparse
import glob
import json
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

from dsp2.env.egcs_env import EGCSEnv
from dsp2.agents.ddqn_agent import DDQNAgent, AgentConfig

import yaml

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
    )
    return env


def _find_latest_ckpt(logdir: str) -> str | None:
    # Prefer explicit final, else latest step
    final = os.path.join(logdir, 'ckpt_final.pt')
    if os.path.exists(final):
        return final
    candidates = glob.glob(os.path.join(logdir, 'ckpt_step_*.pt'))
    if not candidates:
        return None
    def _step(p: str) -> int:
        base = os.path.basename(p)
        try:
            return int(base.split('_')[-1].split('.')[0])
        except Exception:
            return -1
    candidates.sort(key=_step)
    return candidates[-1]


def _load_agent(env: EGCSEnv, ckpt_path: str | None, cfg: dict) -> DDQNAgent:
    agent_cfg = AgentConfig(
        gamma=cfg.get('gamma', 0.99),
        lr=cfg.get('lr', 1e-4),
        batch_size=cfg.get('batch_size', 64),
        target_update_steps=cfg.get('target_update_steps', 10000),
        tau=cfg.get('tau', 1.0),
        epsilon_start=0.0,  # greedy at eval
        epsilon_end=0.0,
        decay_steps=1,
        grad_clip=cfg.get('grad_clip', 5.0)
    )
    agent = DDQNAgent(env.state_size, env.N, env.M, config=agent_cfg)
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            data = torch.load(ckpt_path, map_location='cpu')
            if 'policy' in data:
                agent.policy.load_state_dict(data['policy'])
                agent.target.load_state_dict(agent.policy.state_dict())
        except RuntimeError as e:
            print(f"[WARN] Could not load checkpoint due to shape mismatch: {e}")
    return agent


def evaluate(env: EGCSEnv, agent: DDQNAgent, episodes: int = 5, seed: int | None = None) -> Dict[str, Any]:
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    episode_returns: List[float] = []
    avg_waiting_per_ep: List[float] = []
    avg_incar_per_ep: List[float] = []
    first_ep_traces: Dict[str, List[float]] = {"n_waiting": [], "n_incar": [], "reward": []}

    for ep in range(episodes):
        s = env.reset()
        done = False
        ep_ret = 0.0
        steps = 0
        waiting_acc = 0.0
        incar_acc = 0.0
        while not done:
            mask = env.action_mask()
            a = agent.select_action(s, mask, epsilon=0.0)
            s, r, done, info = env.step(a)
            ep_ret += r
            steps += 1
            waiting_acc += info.get('n_waiting', 0)
            incar_acc += info.get('n_incar', 0)
            if ep == 0:
                first_ep_traces["n_waiting"].append(float(info.get('n_waiting', 0)))
                first_ep_traces["n_incar"].append(float(info.get('n_incar', 0)))
                first_ep_traces["reward"].append(float(r))
        episode_returns.append(ep_ret)
        avg_waiting_per_ep.append(waiting_acc / max(1, steps))
        avg_incar_per_ep.append(incar_acc / max(1, steps))

    metrics = {
        "episodes": episodes,
        "episode_returns": episode_returns,
        "return_mean": float(np.mean(episode_returns)),
        "return_std": float(np.std(episode_returns)),
        "avg_waiting_mean": float(np.mean(avg_waiting_per_ep)),
        "avg_waiting_std": float(np.std(avg_waiting_per_ep)),
        "avg_incar_mean": float(np.mean(avg_incar_per_ep)),
        "avg_incar_std": float(np.std(avg_incar_per_ep)),
        "first_episode_traces": first_ep_traces,
    }
    return metrics


def _load_training_curves(logdir: str) -> Dict[str, List[Tuple[int, float]]]:
    curves: Dict[str, List[Tuple[int, float]]] = {}
    event_files = glob.glob(os.path.join(logdir, 'events.out.tfevents.*'))
    if not event_files:
        return curves
    # Use latest event file
    event_files.sort(key=os.path.getmtime)
    ea = event_accumulator.EventAccumulator(event_files[-1], size_guidance={
        event_accumulator.SCALARS: 100000
    })
    try:
        ea.Reload()
    except Exception as e:
        print(f"[WARN] Failed to parse tensorboard events: {e}")
        return curves
    for tag in ['loss', 'q_mean', 'episode_return', 'n_waiting']:
        if tag in ea.Tags().get('scalars', []):
            scalars = ea.Scalars(tag)
            curves[tag] = [(s.step, s.value) for s in scalars]
    return curves


def save_visualizations(metrics: Dict[str, Any], out_dir: str, training_curves: Dict[str, List[Tuple[int, float]]] | None = None) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []
    # 1) Episode returns bar
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    ax1.bar(np.arange(len(metrics['episode_returns'])), metrics['episode_returns'])
    ax1.set_title(f"Episode returns (mean={metrics['return_mean']:.1f})")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Return")
    fig1.tight_layout()
    path1 = os.path.join(out_dir, 'episode_returns.png')
    fig1.savefig(path1, dpi=150)
    plt.close(fig1)
    paths.append(path1)

    # 2) First episode traces
    traces = metrics.get('first_episode_traces', {})
    fig2, ax2 = plt.subplots(2, 1, figsize=(7, 4), sharex=True)
    ax2[0].plot(traces.get('n_waiting', []), label='n_waiting')
    ax2[0].plot(traces.get('n_incar', []), label='n_incar')
    ax2[0].legend()
    ax2[0].set_ylabel('Count')
    ax2[0].set_title('First episode counts')
    ax2[1].plot(traces.get('reward', []), color='tab:green')
    ax2[1].set_ylabel('Reward')
    ax2[1].set_xlabel('Time step')
    fig2.tight_layout()
    path2 = os.path.join(out_dir, 'first_episode_traces.png')
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)
    paths.append(path2)

    # 3) Training curves if available
    if training_curves:
        for tag, data in training_curves.items():
            if not data:
                continue
            steps, values = zip(*data)
            fig, ax = plt.subplots(figsize=(6,3))
            ax.plot(steps, values)
            ax.set_title(f"Training curve: {tag}")
            ax.set_xlabel('Step')
            ax.set_ylabel(tag)
            fig.tight_layout()
            outp = os.path.join(out_dir, f'train_{tag}.png')
            fig.savefig(outp, dpi=150)
            plt.close(fig)
            paths.append(outp)

    # Save raw metrics
    with open(os.path.join(out_dir, 'eval_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=os.path.join('dsp2', 'configs', 'default.yaml'))
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    logdir = cfg.get('logdir', os.path.join('dsp2', 'logs'))
    ckpt = args.ckpt or _find_latest_ckpt(logdir)

    env = make_env(cfg)
    agent = _load_agent(env, ckpt, cfg)

    metrics = evaluate(env, agent, episodes=args.episodes, seed=args.seed)
    training_curves = _load_training_curves(logdir)
    out_dir = os.path.join(logdir, 'eval')
    paths = save_visualizations(metrics, out_dir, training_curves)

    print(f"Eval complete. mean return={metrics['return_mean']:.3f}, avg waiting={metrics['avg_waiting_mean']:.2f}")
    if ckpt:
        print(f"Used checkpoint: {ckpt}")
    print("Saved plots:")
    for p in paths:
        print(f"  {p}")


if __name__ == '__main__':
    main()
